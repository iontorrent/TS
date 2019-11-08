# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

from __future__ import absolute_import
import re
import os
import sys
import pickle
import stat
import time
import errno
import traceback
import iondb.settings as settings
from iondb.utils.files import percent_full, getdeviceid
from iondb.rundb.models import FileServer, ReportStorage, DMFileSet
from celery.utils.log import get_task_logger

# Send logging to data_management log file
logger = get_task_logger("data_management")
logid = {"logid": "%s" % ("dm_utils")}


def get_walk_filelist(input_dirs, list_dir=None, save_list=False):
    """
    Purpose of the function is to generate a list of all files rooted in the given directories,
    much like os.walk().
    Since the os.walk is an expensive operation on large filesystems, we can store the file list
    in a text file in the report directory.  Once a report is analyzed, the only potential changes
    to the file list will be in the plugin_out directory.  Thus, we always update the contents of
    the file list for that directory.
    """
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    def dump_the_list(filepath, thelist):
        """
        Write a file to report directory containing cached file list.
        Needs to have same uid/gid as directory with 0x666 permissions
        """
        if filepath == "":
            return

        uid = os.stat(os.path.dirname(filepath)).st_uid
        gid = os.stat(os.path.dirname(filepath)).st_gid

        mode = (
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IROTH
            | stat.S_IWOTH
        )  # 0o666
        umask_original = os.umask(0)

        with os.fdopen(os.open(filepath, os.O_WRONLY | os.O_CREAT, mode), "w") as fileh:
            pickle.dump(thelist, fileh)
        os.chown(filepath, uid, gid)
        os.umask(umask_original)

    use_file_cache = False
    # Disable writing the cache file if use_file_cache is False
    if not use_file_cache:
        save_list = False

    thelist = []
    if list_dir:
        cachefile = os.path.join(list_dir, "cached.filelist")
    else:
        cachefile = ""

    starttime = time.time()
    if use_file_cache and os.path.isfile(cachefile):
        # if file of cached list exists, read it
        with open(cachefile, "r") as fileh:
            thelist = pickle.load(fileh)

        # Remove all plugin_out contents from the list
        thelist = [item for item in thelist if "plugin_out" not in item]

        # Update the plugins directory contents now.  We test all entries in input_dirs, not knowing
        # which one contains the plugin_out subdirectory.
        for item in input_dirs:
            try:
                os.chdir(item)
            except OSError as e:
                if e.errno == errno.ENOENT:  # No such file or directory
                    logger.warn("No such directory: %s" % item, extra=logid)
                    continue
                else:
                    logger.error(
                        "Unhandled error in get_walk_filelist on: %s" % item,
                        extra=logid,
                    )
                    logger.error(traceback.format_exc(), extra=logid)
                    continue

            this_dir = os.path.join(item, "plugin_out")
            if os.path.isdir(this_dir):
                os.chdir(this_dir)
                for root, dirs, files in os.walk("./", topdown=True):
                    for j in dirs + files:
                        # This code address specific issue caused by bad plugin code: TS-9917
                        try:
                            j.decode("utf-8")
                        except Exception:
                            logger.warn(
                                "Bad file in directory: %s" % os.path.join(item, root),
                                extra=logid,
                            )
                            logger.warn("File is '%s'" % j, extra=logid)
                            continue
                        this_file = os.path.join(this_dir, root.replace("./", ""), j)
                        if not os.path.isdir(this_file):  # exclude directories
                            thelist.append(this_file)
    else:
        for item in input_dirs:
            try:
                os.chdir(item)
            except OSError as e:
                if e.errno == errno.ENOENT:  # No such file or directory
                    logger.warn("No such directory: %s" % item, extra=logid)
                    continue
                else:
                    logger.error(
                        "Unhandled error in get_walk_filelist on: %s" % item,
                        extra=logid,
                    )
                    logger.error(traceback.format_exc(), extra=logid)
                    continue

            for root, dirs, files in os.walk("./", topdown=True):
                for j in dirs + files:
                    # This code address specific issue caused by bad plugin code: TS-9917
                    try:
                        j.decode("utf-8")
                    except Exception:
                        logger.warn(
                            "Bad file in directory: %s" % os.path.join(item, root),
                            extra=logid,
                        )
                        logger.warn("File is '%s'" % j, extra=logid)
                        continue
                    this_file = os.path.join(item, root.replace("./", ""), j)
                    if not os.path.isdir(this_file):  # exclude directories
                        thelist.append(this_file)
                    elif j == "sigproc_results":
                        # add files from linked sigproc_results folder, except proton onboard_results files
                        if os.path.islink(
                            this_file
                        ) and "onboard_results" not in os.path.realpath(this_file):
                            for root2, _, files2 in os.walk(this_file):
                                thelist += [
                                    os.path.join(root2, filename) for filename in files2
                                ]

    endtime = time.time()
    logger.info(
        "%s: %f seconds" % (sys._getframe().f_code.co_name, (endtime - starttime)),
        extra=logid,
    )

    if save_list and thelist:
        try:
            dump_the_list(cachefile, thelist)
        except Exception:
            # Remove possible partial file
            try:
                os.unlink(cachefile)
            except OSError:
                pass
            logger.error(traceback.format_exc(), extra=logid)

    return thelist


def _file_selector(
    start_dir,
    ipatterns,
    epatterns,
    kpatterns,
    exclude_onboard_results=False,
    add_linked_sigproc=False,
    cached=None,
):
    """Returns list of files found in directory which match the list of
    patterns to include and which do not match any patterns in the list
    of patterns to exclude.  Also returns files matching keep patterns in
    separate list.
    """
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    starttime = time.time()  # debugging time of execution
    to_include = []
    to_exclude = []
    to_keep = []

    if not ipatterns:
        ipatterns = []
    if not epatterns:
        epatterns = []
    if not kpatterns:
        kpatterns = []

    exclude_sigproc_folder = False
    if not add_linked_sigproc and os.path.islink(
        os.path.join(start_dir, "sigproc_results")
    ):
        exclude_sigproc_folder = True

    # find files matching include filters from start_dir
    # for root, dirs, files in os.walk(start_dir,topdown=True):
    for filepath in cached:
        if exclude_onboard_results and "onboard_results" in filepath:
            continue
        if exclude_sigproc_folder and "sigproc_results" in filepath:
            continue

        for pattern in ipatterns:
            file_filter = re.compile(
                r"(%s/)(%s)" % (start_dir, pattern)
            )  # NOTE: use of start_dir, not root here
            match = file_filter.match(filepath)
            if match:
                to_include.append(filepath)

        # find files matching keep filters from start_dir
        for pattern in kpatterns:
            file_filter = re.compile(r"(%s/)(%s)" % (start_dir, pattern))
            match = file_filter.match(filepath)
            if match:
                to_keep.append(filepath)

    # find files matching exclude filters from include list
    for pattern in epatterns:
        file_filter = re.compile(r"(%s/)(%s)" % (start_dir, pattern))
        for filename in to_include:
            match = file_filter.match(filename)
            if match:
                to_exclude.append(filename)

    selected = list(set(to_include) - set(to_exclude))
    endtime = time.time()
    logger.info(
        "%s(): %f seconds" % (sys._getframe().f_code.co_name, (endtime - starttime)),
        extra=logid,
    )
    return selected, to_keep


def dm_category_list():
    """Returns list of fileset cateogries each with list of partition ids that can be acted upon"""

    # Get list of File Server objects
    file_servers = list(FileServer.objects.all().order_by("pk").values())

    # Get list of Report Storage objects
    report_storages = list(ReportStorage.objects.all().order_by("pk").values())

    # dict of fileset cateogries each with list of partition ids that can be acted upon.
    category_list = {}
    # -------------------------------------------------
    # DELETE action only happens if threshold reached
    # -------------------------------------------------
    dmfilesets = list(
        DMFileSet.objects.filter(version=settings.RELVERSION)
        .filter(auto_action="DEL")
        .values()
    )
    for dmfileset in dmfilesets:

        cat_name = slugify(dmfileset["type"])
        category_list[cat_name] = {
            "dmfileset": dmfileset,
            "devlist": [],
            "partitions": [],
        }

        for partition in _partitions(file_servers, report_storages):
            d_loc = {"logid": "%s" % (hex(partition["devid"]))}

            if partition["diskusage"] >= dmfileset["auto_trigger_usage"]:

                logger.info(
                    "%s %.2f%% exceeds %s %.0f%%"
                    % (
                        partition["path"],
                        partition["diskusage"],
                        dmfileset["type"],
                        dmfileset["auto_trigger_usage"],
                    ),
                    extra=d_loc,
                )

                category_list[cat_name]["devlist"].append(partition["devid"])
                category_list[cat_name]["partitions"].append(partition)

            else:

                logger.info(
                    "%s %.2f%% below %s %.0f%%"
                    % (
                        partition["path"],
                        partition["diskusage"],
                        dmfileset["type"],
                        dmfileset["auto_trigger_usage"],
                    ),
                    extra=d_loc,
                )

        # uniquify the deviceid list
        category_list[cat_name]["devlist"] = list(
            set(category_list[cat_name]["devlist"])
        )

    # -------------------------------------------------------------------------------
    # ARCHIVE action happens as soon as grace period has expired (no threshold check)
    # -------------------------------------------------------------------------------
    for dmfileset in list(
        DMFileSet.objects.filter(version=settings.RELVERSION)
        .filter(auto_action="ARC")
        .values()
    ):

        cat_name = slugify(dmfileset["type"])
        category_list[cat_name] = {
            "dmfileset": dmfileset,
            "devlist": [],
            "partitions": [],
        }

        for partition in _partitions(file_servers, report_storages):
            logger.debug(
                "%s %s" % (partition["path"], hex(partition["devid"])), extra=logid
            )
            category_list[cat_name]["devlist"].append(partition["devid"])
            category_list[cat_name]["partitions"].append(partition)

        # uniquify the deviceid list
        category_list[cat_name]["devlist"] = list(
            set(category_list[cat_name]["devlist"])
        )

    return category_list


def _partitions(file_servers, report_storages):
    """Returns list of partitions"""
    partitions = []
    for fileserver in file_servers:
        if os.path.exists(fileserver["filesPrefix"]):
            partitions.append(
                {
                    "path": fileserver["filesPrefix"],
                    "diskusage": fileserver["percentfull"],
                    "devid": getdeviceid(fileserver["filesPrefix"]),
                }
            )
    for reportstorage in report_storages:
        if os.path.exists(reportstorage["dirPath"]):
            partitions.append(
                {
                    "path": reportstorage["dirPath"],
                    "diskusage": reportstorage.get(
                        "percentfull", percent_full(reportstorage["dirPath"])
                    ),
                    "devid": getdeviceid(reportstorage["dirPath"]),
                }
            )
    return partitions


def slugify(something):
    """convert whitespace to hyphen and lower case everything"""
    return re.sub(r"\W+", "-", something.lower())
