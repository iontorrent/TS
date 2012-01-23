#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import glob
import getopt

INFO_TXT = '.info.txt'
REF_LIST_TXT = 'reference_list.txt'
PGM_DIR = '/results/PGM_config/'
LIB_FORMAT = 'tmap-f2'
BASE_DIR = "/results/referenceLibrary/"

help_message = '''
Usage:
    updateref -f "tmap-f1" -p "/mnt/pgm_raw_data_storage/PGM_config"

    -h, --help          display this message
    
    -f, --format        reference library format, i.e. tmap-f1
    -r, --reflib_dir    root directory of reference library
    -p, --pgm_dir       PGM config directory
    -n, --reflist-name  reference list name, i.e. reference_list.txt
'''

def get_list(dir_path):
    """ get list of installed TMAP libraries. Each library has its own
    folder under dir_path"""
    reflist = []
    for basedir, dirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith(INFO_TXT):
                shortname = os.path.basename(basedir)
                reflist.append(shortname)
                break
    return reflist

def write_list(reflist, fname=REF_LIST_TXT, basedir=PGM_DIR):
    """ write out the list as fname under basedir. If basedir is not
    present, it will be created with file permission of 777."""
    if not os.path.isdir(basedir):
        try:
            os.umask(0000)
            os.makedirs(basedir)
        except OSError:
            print("Failed to create %s." %(basedir))
    isOkay = os.access(basedir, os.W_OK)
    if isOkay:
        f = open(os.path.join(basedir, fname), 'w')
        for ref in reflist:
            f.write("%s\n" %(ref))
        os.chmod(f, 0666)
        f.close()

def main(lib_format=LIB_FORMAT, basedir=BASE_DIR, pgmdir=PGM_DIR, fname=REF_LIST_TXT):
    if not os.path.isdir(basedir):
        print("'%s' does not exist." %(basedir))
        sys.exit(0)
    
    dir_path = os.path.join(basedir, lib_format)
    ref_list = get_list(dir_path)
    
    if ref_list:
        print('List of library')
        ref_list.sort()
        for ref in ref_list:
            print('-> %s' %(ref))
        write_list(ref_list, fname, pgmdir)
    else:
        print("No %s library is found under '%s'." %(lib_format, basedir))

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:r:p:n:", 
        ["help", "format=", "reflib-dir=", "pgm-dir=", "reflist-name="])
    except getopt.GetoptError, msg:
        print(str(msg))
        sys.exit(2)

    # option processing
    lib_format = LIB_FORMAT
    reflib_dir = BASE_DIR
    pgm_dir = PGM_DIR
    ref_list_name = REF_LIST_TXT
    for option, value in opts:
        if option in ("-h", "--help"):
            print(help_message)
            sys.exit(0)
        if option in ("-f", "--format"):
            lib_format = value
        if option in ("-r", "--reflib-dir"):
            reflib_dir = value
        if option in ("-p", "--pgm-dir"):
            pgm_dir = value
        if option in ("-n", "--reflist-name"):
            ref_list_name = value
    main(lib_format, reflib_dir, pgm_dir, ref_list_name)
