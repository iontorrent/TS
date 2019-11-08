#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime
import os
import traceback

import zipfile


def make_zip(zip_file, to_zip, arcname=None, use_sys_zip=True, compressed=True):
    """Try to make a zip of a file if it exists
    zip_file is the name of the archive file,
    to_zip is the name of the file to add to the archive,
    arcname is optional; renames the file in the archive,
    use_sys_zip flag will call 'zip' shell command to create archive"""
    # bug in python 2.6 with large zip files; use system zip until its fixed.
    #    printtime("Start make_zip on %s" % to_zip)

    try:
        compression = zipfile.ZIP_DEFLATED
    except Exception:
        compression = zipfile.ZIP_STORED

    if not compressed:
        compression = zipfile.ZIP_STORED
        printtime("not compressed")
    else:
        printtime("compressed")

    if os.path.exists(to_zip):
        if use_sys_zip:
            if arcname != None:
                # This is a hack to trigger only storing of names, no directory heirarchy
                cmd = "zip --junk-paths %s %s" % (zip_file, to_zip)
            else:
                cmd = "zip %s %s" % (zip_file, to_zip)
            try:
                os.system(cmd)
            except Exception:
                printtime("Error executing:\n%s" % cmd)
                print(traceback.format_exc())
        else:
            zf = zipfile.ZipFile(zip_file, mode="a", allowZip64=True)
            try:
                # adding file with compression
                if arcname == None:
                    zf.write(to_zip, compress_type=compression)
                else:
                    zf.write(to_zip, arcname, compress_type=compression)
            #                print "Created ", zip_file, " of", to_zip
            except OSError:
                print("OSError with - :", to_zip)
            except zipfile.LargeZipFile:
                printtime(
                    "The zip file was too large, ZIP64 extensions could not be enabled"
                )
            except:
                printtime("Unexpected error creating zip")
                traceback.print_exc()
            finally:
                zf.close()
    #        printtime("End make_zip %s" % to_zip)
    else:
        printtime(
            "File %s not found.  Zipfile %s not created/updated."
            % (str(to_zip), str(zip_file))
        )
