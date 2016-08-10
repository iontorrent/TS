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


using namespace std;


bool contains(std::string const &fullString, std::string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        std::size_t found = fullString.find(teststring);
        return (found!=std::string::npos);
    }
    else
        return false;
}


string make_acq_name(int i)
{
    char name[MAX_PATH_LENGTH];
    sprintf ( name, "acq_%04d.dat", i );
    string strName = name;
    return strName;
}


string make_prerun_name(int i)
{
    const char *fnames[] = {"beadfind_post_0000.dat", "beadfind_post_0001.dat", "beadfind_post_0002.dat", "beadfind_post_0003.dat",
                               "beadfind_pre_0000.dat", "beadfind_pre_0001.dat", "beadfind_pre_0002.dat", "beadfind_pre_0003.dat",
                               "prerun_0000.dat", "prerun_0001.dat", "prerun_0002.dat", "prerun_0003.dat", "prerun_0004.dat",
                               "prerun_0005.dat", "prerun_0006.dat", "prerun_0007.dat"
                              };
    vector<string> prerun_files(fnames, fnames + sizeof(fnames)/sizeof(fnames[0])); //init vector
    int nFiles = prerun_files.size();
    if (i>=nFiles)
        return ("");
    else
        return (prerun_files[i]);
}


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


int is_multipleOf(int n, int base)
{
    return ((n%base)==0 ? true:false);
}


// thread function declaration
void *do_region_crop ( void *ptr );


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


void crop_image_to_regions(struct foobar &data, struct crop_region CropRegions[], int numRegions)
{
    cout << "crop_image_to_regions..." << endl<< flush;
    pthread_t threads[numRegions];
    for ( int region = 0; region < numRegions;region++ )
    {
    data.CropRegion = &CropRegions[region];
    //data.i = i;
    pthread_create ( &threads[region],NULL, do_region_crop, ( void * ) &data );
    pthread_join ( threads[region],NULL );
    //sequential execution of threads.  Is Image thread safe?
    }
}


bool cropFile_all_regions(const char *expPath, const char *destPath, const char *destName, struct crop_region CropRegions[], int nRegions,bool allocate=true, int doAscii=0, int vfc=1)
{
    string inFile = joinPath(expPath,destName);
    cout<< "cropFile_all_regions... loading " << inFile << endl << flush;
    Image loader;
    if ( loader.LoadRaw_noWait(inFile.c_str(), 0, allocate, false) )
    {
      // create data structure for threads
      Image loader;
      Acq saver;
      const RawImage *raw = loader.GetImage();
      struct foobar data;
      data.CropRegion = NULL;
      data.raw = raw;
      data.destName = strdup ( destName );
      data.destPath = strdup ( destPath );
      data.saver = &saver;
      data.loader = &loader;
      data.doAscii = doAscii;
      data.vfc = vfc;

      pthread_t threads[nRegions];
      for (int region=0; region<nRegions; region++ )
      {
        try
        {
            cout<< "cropFile_all_regions... region " << region << endl << flush;
            data.CropRegion = &CropRegions[region];
            data.i = 0; //i ??
            pthread_create ( &threads[region], NULL, do_region_crop, (void *)&data );
            pthread_join ( threads[region],NULL );
        }
        catch (...)
        {
            cerr <<  "cropFile_all_regions... exception in do_region_crop!!!" << endl << flush;
            exit(1);
        }
      }
      return true;
    }
    else // fail to load
    {
        cout<< "cropFile_all_regions... failed to load " << inFile << endl << flush;
        return false;
    }
}


void crop_prerun_files(const char *expPath, const char *destPath, struct crop_region CropRegions[], int nRegions, bool allocate=true)
{
    const char *fnames[] = {"beadfind_post_0000.dat", "beadfind_post_0001.dat", "beadfind_post_0002.dat", "beadfind_post_0003.dat",
                               "beadfind_pre_0000.dat", "beadfind_pre_0001.dat", "beadfind_pre_0002.dat", "beadfind_pre_0003.dat",
                               "prerun_0000.dat", "prerun_0001.dat", "prerun_0002.dat", "prerun_0003.dat", "prerun_0004.dat",
                               "prerun_0005.dat", "prerun_0006.dat", "prerun_0007.dat"
                              };
    vector<string> prerun_files(fnames, fnames + sizeof(fnames)/sizeof(fnames[0])); //init vector
    int nFiles = prerun_files.size();
    for (int i=0; i<nFiles; i++)
        cropFile_all_regions(expPath,destPath,prerun_files[i].c_str(),CropRegions,nRegions,allocate);
}


void crop_acq_files(const char *expPath, const char *destPath, struct crop_region CropRegions[], int nRegions, int flowlimit=0,bool allocate=true)
{
    int i=0;
    while (true)
    {
        if (flowlimit>0 && i>=flowlimit)
            break;
        char name[MAX_PATH_LENGTH];
        sprintf (name, "acq_%04d.dat", i);
        if (! cropFile_all_regions(expPath,destPath,name,CropRegions,nRegions,allocate)) // no more files to load
            break;
        i++;
    }
}


void usage ( int cropx, int cropy, int kernx, int kerny )
{
  fprintf ( stdout, "CropRegions - Utility to chunk a raw data set into x*y cropped regions.\n" );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "   -a\tOutput flat files; ascii text\n" );
  fprintf ( stdout, "   -b\tUse alternate sampling rate\n" );
  fprintf ( stdout, "   -x\tNumber of blocks along x axis. Default is %d\n",cropx );
  fprintf ( stdout, "   -y\tNumber of blocks along y axis. Default is %d\n",cropy );
  fprintf ( stdout, "   -X\tkernel X size (multiple of 8). Default is %d\n",kernx );
  fprintf ( stdout, "   -Y\tkernel Y size (multiple of 8). Default is %d\n",kerny );
  fprintf ( stdout, "   -t\tchip type 314, 316, or 318\n" );
  fprintf ( stdout, "   -i\tInput directory containing raw data\n" );
  fprintf ( stdout, "   -o\tOutput directory.\n" );
  fprintf ( stdout, "   -f\tConverts only the one file named as an argument\n" );
  fprintf ( stdout, "   -F\tFlowLimit\n" );
  fprintf ( stdout, "   -H\tPrints this message and exits.\n" );
  fprintf ( stdout, "   -v\tPrints version information and exits.\n" );
  fprintf ( stdout, "   -c\tOutput a variable rate frame compressed data set.  Default to whole chip\n" );
  fprintf ( stdout, "   -n\tOutput a non-variable rate frame compressed data set.\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   CropRegions -i /results/analysis/PGM/testRun1 -t [314|316|318]\n" );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}

int main ( int argc, char *argv[] )
{
  int cropx = 2, cropy = 2;
  int kernx = 64, kerny = 64;
  int region_len_x = 0;
  int region_len_y = 0;
  char *expPath  = const_cast<char*> ( "." );
  char *destPath = const_cast<char*> ( "." );
  int chipType = 0;
  string flowrange;
  int flowstart = 0;
  int flowlimit = 0;
  //int alternate_sampling=0;
  int doAscii = 0;
  int vfc = 1;
  //int dont_retry = 0;
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
      flowrange = argv[argcc];
      if (contains(flowrange,"-"))
      {
           vector<string> words;
           split(flowrange,'-',words);
           assert (words.size()==2);
           flowstart = atoi(words[0].c_str());
           flowlimit = atoi(words[1].c_str());
      }
      else
        flowlimit = atoi(argv[argcc]);
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
      //dont_retry = 1;
	  usage ( cropx, cropy, kernx, kerny );
      break;

    case 'c':
      vfc=1;
      cropx=0;
      cropy=0;
      //cropw=0;
      //croph=0;
      break;

    case 'n':
      vfc=0;
      break;

    case 'b':
      //alternate_sampling=1;
      break;

    case 'v':
      fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "CropRegions" ).c_str() );
      exit ( 0 );
      break;

    case 'H':
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
      cout << "Unknown chipType, setting to default 314" << endl << flush;
      chipType = 314;
  }

  string dst = destPath;
  dst += "/CropRegions";
  destPath = const_cast<char*> (dst.c_str());

  // Create results folder
  make_dir(destPath);
  //copy_misc_files(expPath,destPath);   // Copy explog.txt file: all .txt files

  // Initialize array of crop regions
  int numRegions = cropx * cropy;
  //pthread_t threads[numRegions];

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
  } else {
    fprintf ( stderr, "Unknown chip: %d\n", chipType );
    exit ( 1 );
  }

  if (!is_multipleOf(kernx,8))
  {
      int kernx_new = (kernx/8+1)*8;
      printf ( "kernx=%d is not a multiple of 8, changing to %d", kernx, kernx_new);
      fflush ( stdout );
      kernx = kernx_new;
  }

  if (!is_multipleOf(kerny,8))
  {
      int kerny_new = (kerny/8+1)*8;
      printf ( "kerny=%d is not a multiple of 8, changing to %d", kerny, kerny_new);
      fflush ( stdout );
      kerny = kerny_new;
  }

  assert(region_len_x >= kernx);
  assert(region_len_y >= kerny);
  int marginx = (region_len_x - kernx) / 2;
  int marginy = (region_len_y - kerny) / 2;

  //Open outputfile for the BlockStatus text
  FILE *blockline = NULL;
  blockline = fopen ( "blockStatus_output", "w" );

  struct crop_region CropRegions[numRegions];
  for ( int y = 0; y < cropy;y++ ) {
    for ( int x = 0; x < cropx;x++ ) {
      int region = x + y*cropx;
      //CropRegions[region].region_len_x = region_len_x;
      //CropRegions[region].region_len_y = region_len_y;
      //CropRegions[region].region_origin_x = x * region_len_x;
      //CropRegions[region].region_origin_y = y * region_len_y;
      CropRegions[region].region_len_x = kernx;
      CropRegions[region].region_len_y = kerny;
      int newx0 = x * region_len_x + marginx;
      int newy0 = y * region_len_y + marginy;
      if (!is_multipleOf(newx0,8))
      {
          newx0 = (newx0/8+1)*8;
          if (newx0+kernx >= cropx*region_len_x)
              newx0 -= 8;
      }
      if (!is_multipleOf(newy0,8))
      {
          newy0 = (newy0/8+1)*8;
          if (newy0+kerny >= cropy*region_len_y)
              newy0 -= 8;
      }
      CropRegions[region].region_origin_x = newx0;
      CropRegions[region].region_origin_y = newy0;
      snprintf ( CropRegions[region].region_name, 24, "x%04d_y%04d",
                 CropRegions[region].region_origin_x,
                 CropRegions[region].region_origin_y );

      char destSubPath[200];
      sprintf (destSubPath,"%s/%s",destPath,CropRegions[region].region_name);
      make_dir(destSubPath);
      if (flowstart==0)
        copy_misc_files(expPath,destSubPath);

      //write out the BLockStatus line
      fprintf ( blockline, "BlockStatus: X%04d, Y%04d, W%d, H%d, AutoAnalyze:1, AnalyzeEarly:1, nfsCopy:/results-dnas1,  ftpCopy://\n",
                CropRegions[region].region_origin_x,
                CropRegions[region].region_origin_y,
                CropRegions[region].region_len_x,
                CropRegions[region].region_len_y );
    }
  }
  fclose ( blockline );

  //copy_misc_files(expPath,destPath);     // Copy explog.txt file: all .txt files
  //crop_prerun_files(expPath,destPath,CropRegions,numRegions);
  //crop_acq_files(expPath,destPath,CropRegions,numRegions);

  Image loader;
  Acq saver;
  string name, destName;
  bool allocate = true; // this doesn't matter, not used in the functions
  int i = 0;
  // prerun files
  while (true)
  {
    if (flowstart>0)
        break; // crop prerun only when flowstart==0
    if (flowlimit>0 && i>=flowlimit)
        break;

    destName = make_prerun_name(i);
    if (destName.empty())
        break;
    name = joinPath(expPath,destName.c_str());
    if ( loader.LoadRaw_noWait ( name.c_str(), 0, allocate, false ) )
    {
        // create data structure for threads
        struct foobar data;
        data.CropRegion = NULL;
        data.raw = loader.GetImage();
        data.destName = strdup ( destName.c_str() );
        data.destPath = strdup ( destPath );
        data.saver = &saver;
        data.loader = &loader;
        data.doAscii = doAscii;
        data.vfc = vfc;

       crop_image_to_regions(data,CropRegions,numRegions);
       i++;
    }
    else
        break;
  }

  // acq files
  allocate = true;
  i = 0;
  while ( true ) {
    if (flowstart>i)
          continue;
    if (flowlimit>0 && i>=flowlimit)
        break;
    destName = make_acq_name(i);
    name = joinPath(expPath,destName.c_str());
    if ( loader.LoadRaw_noWait ( name.c_str(), 0, allocate, false ) )
    {
        allocate = false; // but not really used??
        // create data structure for threads
        struct foobar data;
        data.CropRegion = NULL;
        data.raw = loader.GetImage();
        data.destName = strdup ( destName.c_str() );
        data.destPath = strdup ( destPath );
        data.saver = &saver;
        data.loader = &loader;
        data.doAscii = doAscii;
        data.vfc = vfc;

       crop_image_to_regions(data,CropRegions,numRegions);
       i++;
    }
    else
        break;
  }

  exit ( 0 );
}


void *do_region_crop (void *ptr)
{
  struct foobar *data = ( struct foobar * ) ptr;
  //double startT = get_time();

  char destSubPath[200];
  char destFile[200];
  sprintf (destSubPath,"%s/%s",data->destPath,data->CropRegion->region_name );
  sprintf (destFile,"%s/%s",destSubPath, data->destName );
  //cout<< "do_region_crop... " << destFile << endl << flush;

  make_dir(destSubPath); // Create results folder

  data->saver->SetData ( data->loader );
  int cropx = data->CropRegion->region_origin_x;
  int cropy = data->CropRegion->region_origin_y;
  int cropw = data->CropRegion->region_len_x;
  int croph = data->CropRegion->region_len_y;

  if ( data->doAscii ) {
    fprintf(stdout, "WriteAscii... %s, x=%d y=%d w=%d h=%d\n", destFile, cropx, cropy, cropw, croph );
    fflush(stdout);
    data->saver->WriteAscii ( destFile, cropx, cropy, cropw, croph );
  } else {
    if ( data->vfc ) {
      fprintf(stdout, "WriteVFC... %s, x=%d y=%d w=%d h=%d\n", destFile, cropx, cropy, cropw, croph );
      fflush(stdout);
      data->saver->WriteVFC ( destFile, cropx, cropy, cropw, croph, false ); // verbose=true/false
    } else {
      fprintf(stdout, "Write... %s, x=%d y=%d w=%d h=%d\n", destFile, cropx, cropy, cropw, croph );
      fflush(stdout);
      data->saver->Write ( destFile, cropx, cropy, cropw, croph );
    }
  }

  /*
  double stopT = get_time();
  printf ( "Done region: %s in %0.2lf sec\n", data->CropRegion->region_name,stopT - startT );
  fflush ( stdout );
  */
  return NULL;
}

