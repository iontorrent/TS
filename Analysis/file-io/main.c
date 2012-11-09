#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "main.h"

extern int 
dat_flow_view_main(int argc, char *argv[]);
extern int 
dat_chip_view_main(int argc, char *argv[]);
extern int
wells_chip_view_main(int argc, char *argv[]);
extern int
wells_mask_view_main(int argc, char *argv[]);
extern int
wells_combine_main(int argc, char *argv[]);
extern int
wells_mask_combine_main(int argc, char *argv[]);
extern int
sff_view_main(int argc, char *argv[]);
extern int
sff_index_create_main(int argc, char *argv[]);
extern int
sff_sort_main(int argc, char *argv[]);
extern int
sff_check_main(int argc, char *argv[]);
extern int
sff_cat_main(int argc, char *argv[]);
extern int
rn_check_main(int argc, char *argv[]);

static int 
usage()
{
  fprintf(stderr, "\n");
  fprintf(stderr, "Program: %s (Tools for ion data)\n", PACKAGE_NAME);
  fprintf(stderr, "Version: %s\n\n", PACKAGE_VERSION);
  fprintf(stderr, "Usage:   %s <command> [options]\n\n", PACKAGE_NAME);
  fprintf(stderr, "Command:\n");
  fprintf(stderr, "         datview        view a DAT file\n");
  fprintf(stderr, "         datcview       view/import a DAT CHIP file\n");
  fprintf(stderr, "         wellsview      view a WELLS file\n");
  fprintf(stderr, "         wellscombine   combine multipe WELLS files\n");
  fprintf(stderr, "         maskcombine    combine multipe MASK files\n");
  fprintf(stderr, "         maskview       view a MASK file\n");
  fprintf(stderr, "         sffview        view a SFF file\n");
  fprintf(stderr, "         sffindex       create an indexed SFF file\n");
  fprintf(stderr, "         sffsort        sort a SFF file\n");
  fprintf(stderr, "         sffcheck       check a SFF file\n");
  fprintf(stderr, "         sffcat         concatentates two or more SFF files\n");
  fprintf(stderr, "         rncheck        check the read name hash\n");
  fprintf(stderr, "\n");
  return 1;
}

int 
main(int argc, char *argv[]) 
{
#ifdef _WIN32
  setmode(fileno(stdout), O_BINARY);
  setmode(fileno(stdin),  O_BINARY);
#endif

  if(argc < 2) return usage();
  else if(0 == strcmp(argv[1], "datview")) return dat_flow_view_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "datcview")) return dat_chip_view_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "wellsview")) return wells_chip_view_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "wellscombine")) return wells_combine_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "maskcombine")) return wells_mask_combine_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "maskview")) return wells_mask_view_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "sffview")) return sff_view_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "sffindex")) return sff_index_create_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "sffsort")) return sff_sort_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "sffcheck")) return sff_check_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "sffcat")) return sff_cat_main(argc-1, argv+1);
  else if(0 == strcmp(argv[1], "rncheck")) return rn_check_main(argc-1, argv+1);
  else {
      fprintf(stderr, "unrecognized command '%s'\n", argv[1]);
      return 1;
  }
  return 0;
}
