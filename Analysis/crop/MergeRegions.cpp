/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <sys/stat.h>

#ifdef __linux__
#include <sys/vfs.h>
#endif

#ifdef __APPLE__
#include <sys/uio.h>
#include <sys/mount.h>
#endif

#include <errno.h>
#include <assert.h>
#include "Image.h"
#include "crop/Acq.h"
#include "IonVersion.h"
#include "Utils.h"

#include "MergeAcq.h"
#include "Acq.h"
#include "OptArgs.h"

#include <list>
#include <iostream>


using namespace std;


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


bool file_exist(string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}


int remove_dir(const char *path, bool verbose=false)
{
    string dir = path;
    string opt = verbose ? " -fvr " : " -fr ";
    string cmd = "exec rm" + opt + dir;
    int rc = system(cmd.c_str());
    return rc;
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


int make_dir(const char *path, mode_t mode=0777)
{
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


bool startswith(string const &fullString, string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        int testlen = teststring.length();
        return (0 == fullString.compare (0, testlen, teststring));
    }
    else
        return false;
}


bool endswith(string const &fullString, string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        int testlen = teststring.length();
        return (0 == fullString.compare (fullString.length() - testlen, testlen, teststring));
    }
    else
        return false;
}

bool contains(string const &fullString, string const &teststring)
{
    if (fullString.length() >= teststring.length()) {
        size_t found = fullString.find(teststring);
        return (found!=string::npos);
    }
    else
        return false;
}


int assignBestX(int xy)
{
    int x = int(sqrt(xy));
    if (x==0)
        x = 1;
    return (x);
}


void splitList(vector<string>&dirs, int nX, vector<vector<string> > &colList)
{
    int nDirs = dirs.size();
    assert(nDirs%nX == 0);
    int nY = nDirs/nX;
    colList.resize(nX);
    int k = 0;
    for (int x=0; x<nX; x++)
    {
        colList[x].resize(nY);
        for (int y=0; y<nY; y++)
        {
            colList[x][y] = dirs[k++];
            cout << "splitList: x=" << x << " y=" << y << " " << colList[x][y] << endl << flush;
        }
    }
}


int readList(char *listPath, vector<string> &lines, string commentChar = "#")
{
    lines.resize(0);
    string line;
    ifstream textFile (listPath);
    if (textFile.is_open())
    {
        while (textFile.good())
        {
            getline(textFile,line);
            if ((! startswith(line,commentChar)) && line.size()>0)
                lines.push_back(line);
        }
        textFile.close();
    }
    else
    {
        cerr << "Unable to open file " << listPath << endl << flush;
        exit(1);
    }
    return (lines.size());
}


void usage() {
    fprintf (stdout, "MergeRegions - Utility to merge (NxM) cropped regions into a bigger one\n");
    fprintf (stdout, "options:\n");
    fprintf (stdout, "   -l\tList of directories containing the cropped regions \n");
    fprintf (stdout, "   -i\tInput directory containing the explog_final.txt file\n");
    fprintf (stdout, "   -o\tOutput directory.\n");
    fprintf (stdout, "   -F\tFlowLimit\n" );
    fprintf (stdout, "   -z\tTells the image loader not to wait for a non-existent file\n");
    fprintf (stdout, "   -H\tPrints this message and exits.\n");
    fprintf (stdout, "   -v\tPrints version information and exits.\n");
    fprintf (stdout, "   -c\tTells image loader to pay attention to checksums.\n");
    fprintf (stdout, "\n");
    fprintf (stdout, "usage:\n");
    fprintf (stdout, "   MergeRegions -l src.list -o /tmp/ \n");
    fprintf (stdout, "\n");
    exit (1);
}


void merge(MergeAcq* merger, Acq* acq, Image* combo, Image* top, Image* bottom, char* destName){
    double startT = get_time();
    merger->SetFirstImage(bottom);
    merger->SetSecondImage(top, bottom->GetRows(), 0); // starting vertically raised but columns the same.
    //cout << "Merging." << endl;
    merger->Merge(*combo);
    //cout << "Saving. " << endl;
    acq->SetData(combo);
    acq->WriteVFC(destName, 0, 0, combo->GetCols(), combo->GetRows());
    //cout << "Done." << endl;
    double stopT = get_time();
    printf("merge: %s in %0.2lf sec\n\n", destName,stopT - startT);
}


void merge_stack(Image* combo, vector<Image *> &imgStack, const char* destName, bool colWise, bool verbose=false)
{
    //double startT = get_time();
    MergeAcq merger;
    Acq acq;
    RawImage *raw = new RawImage;
    if (raw)
    {
        merger.MergeImageStack(raw,imgStack,colWise);
        combo->SetImage(raw);
        acq.SetData(combo);
        acq.WriteVFC(destName, 0, 0, combo->GetCols(), combo->GetRows(), verbose); // verbose=false
        //double stopT = get_time();
        int rows = combo->GetRows();
        int cols = combo->GetCols();
        if (verbose)
        {
            cout << "merge_stack: merging " << imgStack.size() << " files into " << destName << " rows=" << rows << " cols=" << cols << endl <<flush;
        }
        else
        {
            cout << "merge_stack: " << destName << " rows=" << rows << " cols=" << cols << endl << flush;
            //printf("merge_stack: %s in %0.2lf sec\n\n", destName,stopT - startT);
        }
    }
    else
    {
        cerr << "merge_stack error: raw image allocation error... exiting" << endl << flush;
        exit(1);
    }

}


void merge_images_1D(vector<string> &dirs, Image *combo,int file_type,char *destPath, int dont_retry,int ignoreChecksumErrors,int flowstart,int flowlimit, bool colWise, bool verbose=true)
{
    if (flowstart>0 && file_type<2) // do nothing for prerun files if flowstart>0
        return;
    int nDirs = dirs.size();
    //cout << "merge_images_1D merging " << nDirs << " directories" <<endl << endl;
    bool allocate = true;
    vector<Image *> imgStack(nDirs);
    for (int i=0; i<nDirs; i++)
    {
        imgStack[i] = new Image;
        //cout << "merge_images_1D imgStack[" << i << "]=" << imgStack[i]  << endl << flush;
    }

    int k = 0;
    while (true)
    {
        if (flowstart>k)
            continue;
        if (flowlimit>0 && k>=flowlimit)
            break;
        int nFiles = 0;
        for (int i=0; i<nDirs; i++)
        {
            string name1 = make_image_filename(dirs[i].c_str(),k,file_type);
            if (! file_exist(name1))
            {
                //cout << "file " << name1 << " does not exist, merge_images_1D() terminated at flow " << k << endl << flush;
                break;
            }
            imgStack[i]->SetImgLoadImmediate(false);
            imgStack[i]->SetIgnoreChecksumErrors(ignoreChecksumErrors);
            if (dont_retry)
                imgStack[i]->SetTimeout(1,1); // if requested...do not bother waiting for the files to show up
            else
                imgStack[i]->SetTimeout(5,300); // wait 300 sec at 5 sec interval

            //cout << "merge_images_1D loading image " << name1 << endl << endl;
            //if (verbose) cout << "i=" << i << " LoadRaw_noWait()..." << name1 << endl << flush;
            //allocate = true;
            //allocate = (k>0) ? false : true;
            //allocate = (k>0 && file_type==2) ? false : true;
            //if (! imgStack[i]->LoadRaw(name1.c_str(), 0, allocate, false))
            if (! imgStack[i]->LoadRaw_noWait(name1.c_str(), 0, allocate, false))
            {
                cerr << "cannot load image file: " << name1 << endl << flush;
                exit(1);
                //break;
            }
            nFiles++;
        }

        if (nFiles==nDirs)
        {
        string destName = make_image_filename(destPath,k,file_type);
        merge_stack(combo,imgStack,destName.c_str(),colWise,verbose);
        //cout << "done merge_stack... " << endl << endl;
        }
        else
        {
           if (nFiles>0)
               cerr << "merge_images_1D warning: " << nFiles << " read, less than nDirs(" << nDirs << ")...exiting" << endl << flush;
           break;
        }
        k++;
    }
    for (int i=0; i<nDirs; i++)
        if (imgStack[i])
        {
            delete imgStack[i];
        }

}



void merge_images_2D(vector<string> &dirs, int nX, int file_type,char *expPath,char *destPath, int dont_retry, int ignoreChecksumErrors, int flowstart=0, int flowlimit=0, bool verbose=false)
{
    char destDir[MAX_PATH_LENGTH];
    int nDirs = dirs.size();
    int nY = nDirs/nX;
    bool colWise = true;

    vector<vector<string> > colList;
    splitList(dirs,nX,colList);
    assert (nX>1);
    assert (nY>1 && nY>=nX);

    Image combo;
    //vector<RawImage> combo_col(nX);
    vector<string> col_dirs(nX);
    for (int x=0; x<nX; x++)
    {
        cout << "merge_images_2D... merging " << nY << " regions into one at column " << x << endl << flush;
        sprintf(destDir, "%s/col_%d", destPath, x);
        col_dirs[x] = destDir;
        //remove_dir(destDir); // there are more than one data types, cannot remove_dir here
        //make_dir(destDir);
        //copy_misc_files(expPath,destDir);

        vector<string> column = colList[x];
        merge_images_1D(column,&combo,file_type,destDir,dont_retry,ignoreChecksumErrors,flowstart,flowlimit,colWise,verbose);
    }

    // merge combo_col row-wise into one final region
    cout << "merge_images_2D... merging " << nX << " column strips into the final thumbnail" << endl << flush;
    sprintf(destDir, "%s/final_thumbnail", destPath);
    //remove_dir(destDir);
    //make_dir(destDir);
    //copy_misc_files(expPath,destDir);
    colWise = false;
    merge_images_1D(col_dirs,&combo,file_type,destDir,dont_retry,ignoreChecksumErrors,flowstart,flowlimit,colWise,verbose);
}


void make_dirs_2D(vector<string> &dirs, int nX, char *expPath,char *destPath)
{
    char destDir[MAX_PATH_LENGTH];
    int nDirs = dirs.size();
    int nY = nDirs/nX;

    vector<vector<string> > colList;
    splitList(dirs,nX,colList);
    assert (nY>1 && nY>=nX);

    for (int x=0; x<nX; x++)
    {
        sprintf(destDir, "%s/col_%d", destPath, x);
        remove_dir(destDir); // there are more than one data types, cannot remove_dir here
        make_dir(destDir);
        copy_misc_files(expPath,destDir);
    }

    // merge combo_col row-wise into one final region
    sprintf(destDir, "%s/final_thumbnail", destPath);
    remove_dir(destDir);
    make_dir(destDir);
    copy_misc_files(expPath,destDir);
}



int main(int argc, char *argv[])
{
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    cout << "Hostname: " << hostname << endl;

    time_t rawtime;
    struct tm * timeinfo;
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    cout << "Current local time and date: " << asctime (timeinfo) << endl;

    char *expList = const_cast<char*>(".");
    char *expPath = const_cast<char*>(".");
    char *destPath = const_cast<char*>(".");
    string flowrange;
    int flowstart = 0;
    int flowlimit = 0;
    int dont_retry = 0;
    int ignoreChecksumErrors = 1;
    int nX = 0;

    if (argc <= 2) {
        usage();
    }
    int argcc = 1;
    while (argcc < argc) {
        switch (argv[argcc][1]) {

        case 'l':
            argcc++;
            expList = argv[argcc];
            break;

        case 'x':
            nX = atoi(argv[argcc]);
            break;

        case 'i':
            argcc++;
            expPath = argv[argcc];
            break;

        case 'o':
            argcc++;
            destPath = argv[argcc];
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

        case 'c':
            ignoreChecksumErrors = 0;
            break;

        case 'z':
            dont_retry = 1;
            break;

        case 'v':
            fprintf (stdout, "%s", IonVersion::GetFullVersion("MergeRegions").c_str());
            exit (0);
            break;
        case 'H':
            usage();
            break;

        default:
            argcc++;
            fprintf (stdout, "\n");

        }
        argcc++;
    }


    string dst = destPath;
    dst += "/MergeRegions";
    destPath = const_cast<char*> (dst.c_str());
    // Create results folder
    make_dir(destPath);

    // merge beadfinds as soon as acq_0000.dat is available
    //wait_for_first_acq(expPath1,expPath2);

    vector<string> dirs;
    int nDirs = readList(expList,dirs);
    if (nDirs < 2)
    {
        cerr <<"Error: not enough (2) directories in the file" << endl << flush;
        exit(1);
    }

    // assign proper nX/nY, nX<=sqrt(nDirs)
    if (nX==0)
        nX = assignBestX(nDirs);

    if (nDirs%nX != 0)
    {
        cerr <<"Error: nDirs=" << nDirs << " is not an integer of nX=" << nX << endl << flush;
        exit(1);
    }

    make_dirs_2D(dirs,nX,expPath,destPath); // make subdirs only once
    merge_images_2D(dirs,nX,0,expPath,destPath,dont_retry,ignoreChecksumErrors,flowstart,flowlimit); // file_type=0, beadfind
    merge_images_2D(dirs,nX,1,expPath,destPath,dont_retry,ignoreChecksumErrors,flowstart,flowlimit); // file_type=1, prerun
    merge_images_2D(dirs,nX,2,expPath,destPath,dont_retry,ignoreChecksumErrors,flowstart,flowlimit); // file_type=2, acq

    return EXIT_SUCCESS;
}
