# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Tasks
=====

The ``tasks`` module contains all the Python functions which spawn Celery
tasks in the background.

Not all functions contained in ``tasks`` are actual Celery tasks, only those
that have the  ``@app.task`` decorator.
"""

from __future__ import division, absolute_import

from celery import task, group, chord
from celery.task import periodic_task
from celery.utils.log import get_task_logger
from celery.schedules import crontab
from celery.exceptions import SoftTimeLimitExceeded
from django.core import mail
from django.contrib.auth.models import User
from iondb.celery import app
import urllib2
import os
import signal
import string
import random
import subprocess
import shutil
import socket
from django.conf import settings
from django.utils import timezone
import zipfile
import os.path
import sys
import re
import json
import logging
from datetime import timedelta, datetime
import pytz
import time
import tempfile
import urllib
import traceback
import requests
import feedparser
import dateutil
import urlparse
from iondb.utils import raid as raid_utils
from iondb.utils import files as file_utils
from iondb.rundb.configure.cluster_info import connect_nodetest, config_nodetest, queue_info, sge_ctrl

logger = get_task_logger(__name__)


def call(*cmd, **kwargs):
    if "stdout" not in kwargs:
        kwargs["stdout"] = subprocess.PIPE
    if "stderr" not in kwargs:
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.Popen(cmd, **kwargs)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def run_as_daemon(callback, *args, **kwargs):
    """Disk And Execution MONitor (Daemon)
    Fork off a completely separate process and run callback from that process.
    """
    # fork the first time (to make a non-session-leader child process)
    try:
        pid = os.fork()
    except OSError, e:
        raise RuntimeError("1st fork failed: %s [%d]" % (e.strerror, e.errno))
    if pid != 0:
        # parent (calling) process is all done
        return
    # detach from controlling terminal (to make child a session-leader)
    os.setsid()
    try:
        pid = os.fork()
    except OSError, e:
        raise RuntimeError("2nd fork failed: %s [%d]" % (e.strerror, e.errno))
    if pid != 0:
        # child process is all done
        os._exit(0)
    # grandchild process now non-session-leader, detached from parent
    # grandchild process must now close all open files
    try:
        maxfd = os.sysconf("SC_OPEN_MAX")
    except (AttributeError, ValueError):
        maxfd = 1024

    for fd in range(maxfd):
        try:
            os.close(fd)
        except OSError: # ERROR, fd wasn't open to begin with (ignored)
            pass
    # redirect stdin, stdout and stderr to /dev/null
    os.open(os.devnull, os.O_RDWR) # standard input (0)
    os.dup2(0, 1)
    os.dup2(0, 2)
    # Run our callback function with it's arguments
    callback(*args, **kwargs)
    sys.exit()

# ZipFile doesn't provide a context manager until 2.7/3.2
if hasattr(zipfile.ZipFile, '__exit__'):
    ZipContextManager = zipfile.ZipFile
else:
    class ZipContextManager():

        def __init__(self, *args, **kwargs):
            self.zobj = zipfile.ZipFile(*args, **kwargs)

        def __enter__(self):
            return self.zobj

        def __exit__(self, type, value, traceback):
            self.zobj.close()

# Unified unzip function
def extract_zip(archive, dest, prefix=None, auto_prefix=False, logger=None):
    """ unzip files in archive to destination folder
    extracting only files in prefix and omitting prefix from output path.
    """
    if not logger:
        logger = logging.getLogger(__name__)
    # Normalize, clear out or create dest path
    dest = os.path.normpath(dest)
    if os.path.exists(dest):
        if not os.path.isdir(dest):
            raise OSError("Must extract zip file to a directory. File already exists: '%s'" % dest)

        if dest.find(settings.PLUGIN_PATH) == 0:
            ## Only delete content under PLUGIN_PATH.
            delete_that_folder(dest, "Deleting content at destination path '%s'" % dest)
        elif os.listdir(dest):
            # if the directory is not empty then a error needs to be raised
            raise OSError("Unable to extract ZIP - directory '%s' already exists" % dest)

        os.chmod(dest, 0777)
    else:
        os.makedirs(dest, 0777)

    logger.info("Extracting ZIP '%s' to '%s'", archive, dest)

    try:
        import pwd, grp
        uid = pwd.getpwnam('ionadmin')[2]
        gid = grp.getgrnam('ionadmin')[2]
    except OSError:
        uid = os.getuid()
        gid = os.getgid()

    extracted_files = []
    with ZipContextManager(archive, 'r') as zfobj:
        ## prefix is a string to extract from zipfile
        offset = 0
        if auto_prefix and not prefix:
            prefix, _ = file_utils.get_common_prefix(zfobj.namelist())
        if prefix is not None:
            offset = len(prefix) + 1
            logger.debug("ZIP extract prefix '%s'", prefix)

        for member in zfobj.infolist():
            if member.filename[0] == '/':
                filename = member.filename[1:]
            else:
                filename = member.filename

            if prefix:
                if filename.startswith(prefix):
                    logger.debug("Extracting '%s' as '%s'", filename, filename[offset:])
                    #filename = filename[offset:]
                else:
                    logging.debug("Skipping file outside '%s' prefix: '%s'", filename, prefix)
                    continue

            targetpath = os.path.join(dest, filename)
            targetpath = os.path.normpath(targetpath)

            # Catch files we can't handle properly.
            if targetpath.find(dest) != 0:
                ## Path is no longer under dest after normalization. Prevent extraction (eg. ../../../etc/passwd)
                logging.error("ZIP archive contains file '%s' outside destination path: '%s'. Skipping.", filename, dest)
                continue

            # ZIP archives can have symlinks. Nope.
            if ((member.external_attr << 16L) & 0120000):
                logging.error("ZIP archive contains symlink: '%s'. Skipping.", member.filename)
                continue

            if "__MACOSX" in filename:
                logging.warn("ZIP archive contains __MACOSX meta folder. Skipping %s", member.filename)
                continue

            # Get permission set inside archive
            perm = ((member.external_attr >> 16L) & 0777) or 0755

            # Create all upper directories if necessary.
            upperdirs = os.path.dirname(targetpath)
            if upperdirs and not os.path.exists(upperdirs):
                logger.debug("Creating tree for '%s'", upperdirs)
                os.makedirs(upperdirs, perm | 0555)

            if filename[-1] == '/':
                # upper bits of external_attr should be 04 for folders... ignoring this for now
                if not os.path.isdir(targetpath):
                    logger.debug("ZIP extract dir: '%s'", targetpath)
                    os.mkdir(targetpath, perm | 0555)
                continue

            try:
                with os.fdopen(os.open(targetpath, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, perm), 'wb') as targetfh:
                    zipfh = zfobj.open(member)
                    shutil.copyfileobj(zipfh, targetfh)
                    zipfh.close()
                logger.debug("ZIP extract file: '%s' to '%s'", filename, targetpath)
            except (OSError, IOError):
                logger.exception("Failed to extract '%s':'%s' to '%s'", archive, filename, targetpath)
                continue
            # Set folder or file last modified time (ctime) to date of file in archive.
            try:
                #os.utime(targetpath, member.date_time)
                os.chown(targetpath, uid, gid)
            except (OSError, IOError) as e:
                # Non fatal if time and owner fail.
                logger.warn("Failed to set owner attributes on '%s': %s", targetpath, e)

            extracted_files.append(targetpath)

    return (prefix, extracted_files)


@app.task
def echo(message, wait=0.0):
    time.sleep(wait)
    logger.info("Logged: " + message)
    print(message)


@app.task
def delete_that_folder(directory, message):
    def delete_error(func, path, info):
        logger.error("Failed to delete %s: %s: %s", path, message, info)

    if os.path.exists(directory):
        logger.info("Deleting %s", directory)
        shutil.rmtree(directory, onerror=delete_error)

#N.B. Run as celery task because celery runs with root permissions


@app.task
def removeDirContents(folder_path):
    for file_object in os.listdir(folder_path):
        file_object_path = os.path.join(folder_path, file_object)
        if os.path.isfile(file_object_path):
            os.unlink(file_object_path)
        elif os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)


def downloadChunks(url):
    """Helper to download large files"""

    baseFile = os.path.basename(url)
    uuid_path = ''.join([random.choice(string.letters + string.digits) for i in range(10)])

    #move the file to a more uniq path
    os.umask(0002)
    temp_path = settings.TEMP_PATH
    temp_path_uniq = os.path.join(temp_path, uuid_path)
    os.mkdir(temp_path_uniq)

    try:
        file = os.path.join(temp_path_uniq, baseFile)

        req = urllib2.urlopen(url)
        try:
            total_size = int(req.info().getheader('Content-Length').strip())
        except:
            total_size = 0
        downloaded = 0
        CHUNK = 256 * 10240
        with open(file, 'wb') as fp:
            shutil.copyfileobj(req, fp, CHUNK)
        url = req.geturl()
    except urllib2.HTTPError as e:
        logger.error("HTTP Error: %d '%s'", e.code, url)
        delete_that_folder(temp_path_uniq, "after download error")
        return False
    except urllib2.URLError, e:
        logger.error("URL Error: %s '%s'", e.reason, url)
        delete_that_folder(temp_path_uniq, "after download error")
        return False
    except:
        logger.exception("Other error downloading from '%s'", url)
        delete_that_folder(temp_path_uniq, "after download error")
        return False

    return file, url


@app.task
def downloadGenome(url, genomeID):
    """download a genome, and update the genome model"""
    downloadChunks(url)


@app.task
def downloadPublisher(url, zip_file=None):
    from iondb.rundb.models import Message

    #normalise the URL
    if not zip_file:
        url = urlparse.urlsplit(url).geturl()
        ret = downloadChunks(url)
        if not ret:
            return False
        (zip_file, url) = ret
        pub_name = os.path.splitext(os.path.basename(url))[0]
    else:
        pub_name = os.path.splitext(os.path.basename(zip_file))[0]

    # Extract zipfile - yes, plugins scratch folder, not publisher specific.
    scratch_path = os.path.join(settings.PLUGIN_PATH, "scratch", "publisher-temp", pub_name)
    zip_file = os.path.join(settings.PLUGIN_PATH, "scratch", zip_file)
    try:
        (prefix, files) = extract_zip(zip_file, scratch_path, auto_prefix=True, logger=logger)
        if prefix:
            # Good - ZIP has top level folder with publisher name, use that name instead.
            pub_name = os.path.basename(prefix)
            base_path = os.path.join(scratch_path, pub_name)
        else:
            base_path = scratch_path

        # make sure we have a publisher_meta.json file
        if not os.path.exists(os.path.join(base_path, 'publisher_meta.json')):
            raise Exception('Missing publisher_meta.json!')

        # Move from temp folder into publisher
        pub_final_path = os.path.join("/results/publishers/", pub_name)
        if os.path.exists(pub_final_path):
            # existing publisher will be replaced
            delete_that_folder(pub_final_path, "Error removing old copy of publisher at '%s'" % pub_final_path)
        shutil.move(base_path, pub_final_path)

        ## Rescan Publishers to complete install
        from iondb.rundb import publishers
        publishers.search_for_publishers()

        msg = "Successfully downloaded and installed publisher %s from ZIP archive" % (pub_name,)
        logger.info(msg)
        Message.success(msg)
    except Exception as err:
        msg = "Failed to install publisher from %s. ERROR: %s" % (zip_file, err)
        Message.error(msg)
        logger.exception(msg)
    finally:
        #remove the zip file
        os.unlink(zip_file)
        delete_that_folder(scratch_path, "Error deleting temp publisher zip folder at '%s'" % scratch_path)

    return


@app.task
def contact_info_flyaway():
    """This invokes an external on the path which performs 3 important steps:
        Pull contact information from the DB
        Black magic
        Axeda has the contact information
    """
    logger.info("The user updated their contact information.")
    cmd = ["/opt/ion/RSM/updateContactInfo.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("updateContactInfo.py output error information:\n%s" % stderr)
    return stdout

@app.task
def updateOneTouch():
    sys.path.append("/opt/ion/onetouch")
    from onetouch import findHosts

    #find onetouches
    if os.path.exists("/tmp/OTlock"):
        #remove the OTstatus file if it exists
        if os.path.exists("/tmp/OTstatus"):
            os.unlink("/tmp/OTstatus")
        #touch the status file
        otStatus = open("/tmp/OTstatus", 'w').close()
        #run the onetouch update script
        try:
            updateStatus = findHosts.findOneTouches()
        except:
            updateStatus = "FAILED"
        otStatus = open("/tmp/OTstatus", 'w')
        otStatus.write(str(updateStatus) + "\n")
        otStatus.write("DONE\n")
        otStatus.close()
        #now remove the lock
        os.unlink("/tmp/OTlock")
        return True

    return False


@app.task(queue="slowlane")
def build_tmap_index(reference_id, fasta=None, reference_mask_path=None):
    """ Provides a way to kick off the tmap index generation
        this should spawn a process that calls the build_genome_index.pl script
        it may take up to 3 hours.
        The django server should contacts this method from a view function
        When the index creation processes has exited, cleanly or other wise
        a callback will post to a url that will update the record for the library data
        letting the genome manager know that this now exists
        until then this genome will be listed in a unfinished state.
    """
    from iondb.rundb import models
    reference = models.ReferenceGenome.objects.get(pk=reference_id)
    reference.status = "indexing"
    reference.save()

    if not fasta:
        fasta = os.path.join(reference.reference_path, reference.short_name + ".fasta")
    logger.debug("TMAP %s rebuild, for reference %s(%d) using fasta %s" %
         (settings.TMAP_VERSION, reference.short_name, reference.pk, fasta))

    cmd = [
        '/usr/local/bin/build_genome_index.pl',
        "--auto-fix",
        "--fasta", fasta,
        "--genome-name-short", reference.short_name,
        "--genome-name-long", reference.name,
        "--genome-version", reference.version
    ]
    if reference_mask_path:
        cmd.append("--reference-mask")
        cmd.append(reference_mask_path)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=settings.TMAP_DIR, preexec_fn=os.setsid)

    def termiante_handler(signal, frame):
        if proc.poll() is None:
            proc.terminate()
        raise SystemExit
    signal.signal(signal.SIGTERM, termiante_handler)
    stdout, stderr = proc.communicate()
    ret = proc.returncode

    if ret == 0:
        logger.debug("Successfully built the TMAP %s index for %s" %
                    (settings.TMAP_VERSION, reference.short_name))
        reference.status = 'complete'
        reference.index_version = settings.TMAP_VERSION
        reference.save()
        reference.enable_genome()
    else:
        logger.error('TMAP index rebuild "%s" failed:\n%s' %
                     (" ".join(cmd), stderr))
        reference.status = 'error'
        reference.verbose_error = stderr[:3000]
        reference.save()
        reference.disable_genome()

    return ret == 0


def IonReporterWorkflows():

    try:
        from iondb.rundb import models
        IonReporterUploader = models.Plugin.objects.get(name="IonReporterUploader_V1_0", selected=True, active=True)

        logging.error(IonReporterUploader.config)
        config = IonReporterUploader.config
    except models.Plugin.DoesNotExist:
        error = "IonReporterUploader V1.0 Plugin Not Found."
        logging.error(error)
        return False, error

    try:
        headers = {"Authorization": config["token"]}
        url = config["protocol"] + "://" + config["server"] + ":" + config["port"] + "/grws/analysis/wflist"
        logging.info(url)
    except KeyError:
        error = "IonReporterUploader V1.0 Plugin Config is missing needed data."
        logging.exception(error)
        return False, error

    try:
        #using urllib2 right now because it does NOT verify SSL certs
        req = urllib2.Request(url=url, headers=headers)
        response = urllib2.urlopen(req)
        content = response.read()
        content = json.loads(content)
        workflows = content["workflows"]
        return True, workflows
    except urllib2.HTTPError, e:
        error = "IonReporterUploader V1.0 could not contact the server."
        content = e.read()
        logging.error("Error: %s\n%s", error, content)
        return False, error
    except:
        error = "IonReporterUploader V1.0 could not contact the server."
        logging.exception(error)
        return False, error


def IonReporterVersion(plugin):
    """
    This is a temp thing for 3.0. We need a way for IRU to get the versions
    this will do that for us.
    """

    #if version is pased in use that plugin name instead
    if not plugin:
        plugin = "IonReporterUploader"

    try:
        from iondb.rundb import models
        IonReporterUploader = models.Plugin.objects.get(name=plugin, selected=True, active=True)
        logging.error(IonReporterUploader.config)
        config = IonReporterUploader.config
    except models.Plugin.DoesNotExist:
        error = plugin + " Plugin Not Found."
        logging.exception(error)
        return False, error

    try:
        headers = {"Authorization": config["token"]}
        url = config["protocol"] + "://" + config["server"] + ":" + config["port"] + "/grws_1_2/data/versionList"
        logging.info(url)
    except KeyError:
        error = plugin + " Plugin Config is missing needed data."
        logging.debug(plugin + " config: " + config)
        logging.exception(error)
        return False, error

    try:
        #using urllib2 right now because it does NOT verify SSL certs
        req = urllib2.Request(url=url, headers=headers)
        response = urllib2.urlopen(req)
        content = response.read()
        content = json.loads(content)
        versions = content["Version List"]
        return True, versions
    except urllib2.HTTPError, e:
        error = plugin + " could not contact the server. No versions will be returned"
        content = e.read()
        logging.error("Error: %s\n%s", error, content)
    except:
        error = plugin + " could not contact the server. No versions will be returned"
        logging.exception(error)
        return False, error


@periodic_task(run_every=timedelta(days=1), expires=600, queue="periodic")
def diskspace_status():
    '''Record disk space in a file for historical data
       For every entry in File Servers and Report Storage'''
    from iondb.rundb import models
    import datetime as dt
    directories = []
    newlines = []
    try:
        for repstor in models.ReportStorage.objects.all():
            directories.append(repstor.dirPath)
    except:
        pass
    try:
        for filestor in models.FileServer.objects.all():
            directories.append(filestor.filesPrefix)
    except:
        pass
    for entry in directories:
        q = subprocess.Popen(["df", entry], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = q.communicate()
        if q.returncode == 0:
            device, size, used, available, _, _ = stdout.split("\n")[1].split()
            newlines.append("%s,%s,%s,%s,%s" % (dt.date.today(), device, size, used, available))
        else:
            newlines.append("%s,%s" % (dt.date.today(), stderr))
    with open("/var/log/ion/diskspace.log", "ab") as fh:
        for entry in list(set(newlines)):
            fh.write(entry + "\n")


@periodic_task(run_every=timedelta(days=1), expires=600, queue="periodic")
def scheduled_update_check():
    from iondb.rundb import models
    try:
        packages = check_updates()
        upgrade_message = models.Message.objects.filter(tags__contains="new-upgrade")
        if packages:
            if not upgrade_message.all():
                models.Message.info('There is an update available for your Torrent Server. <a class="btn btn-success" href="/admin/update">Update Now</a>', tags='new-upgrade', route=models.Message.USER_STAFF)
            download_now = models.GlobalConfig.get().enable_auto_pkg_dl
            if download_now:
                async = download_updates.delay()
                logger.debug("Auto starting download of %d packages in task %s" % (packages, async.task_id))
        else:
            upgrade_message.delete()
    except Exception as err:
        logger.error("TSconfig raised '%s' during a scheduled update check." % err)
        models.GlobalConfig.objects.update(ts_update_status="Update failure")
        raise


@app.task
def check_updates():
    """Currently this is passed a TSConfig object; however, there might be a
    smoother design for this control flow.
    """
    from iondb.rundb import models
    from iondb.utils.files import rename_extension
    try:
        from iondb.utils.usb_check import getUSBinstallerpath, change_list_files

        path = getUSBinstallerpath()
        if path:
            change_list_files(path)
        if not path:
            rename_extension('etc/apt/', '.USBinstaller', '')
            if os.path.isfile('/etc/apt/sources.list.d/usb.list'):
                os.remove('/etc/apt/sources.list.d/usb.list')

        cmd = ['sudo', '/usr/lib/python2.7/dist-packages/ion_tsconfig/TSconfig.py', '--poll']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise Exception(stderr)
        else:
            return int(stdout)
    except Exception as err:
        rename_extension('etc/apt/', '.USBinstaller', '')
        if os.path.isfile('/etc/apt/sources.list.d/usb.list'):
            os.remove('/etc/apt/sources.list.d/usb.list')
        logger.error("TSConfig raised '%s' during update check." % err)
        models.GlobalConfig.objects.update(ts_update_status=str(err))
        raise


@app.task
def download_updates(auto_install=False):
    """ Downloads new packages
        If auto_install=True:
            updates ion-tsconfig package then starts software upgrade in a separate process
    """
    from iondb.rundb import models

    try:
        cmd = ['sudo', '/usr/lib/python2.7/dist-packages/ion_tsconfig/TSconfig.py', '--download']
        if auto_install:
            cmd += ['--refresh']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise Exception(stderr)
    except Exception as err:
        logger.error("TSConfig raised '%s' during a download" % err)
        models.GlobalConfig.objects.update(ts_update_status=str(err))
        raise

    if auto_install:
        async = install_updates.delay()
        logger.debug("Auto starting software upgrade in task %s" % async.task_id)


def _do_the_install():
    """This function is expected to be run from a daemonized process"""
    from iondb.rundb import models
    try:
        cmd = ['sudo', '/usr/lib/python2.7/dist-packages/ion_tsconfig/TSconfig.py', '--upgrade']
        p = subprocess.check_call(cmd)
        logger.info('TSConfig install success!')
        success = True
    except Exception as err:
        success = False
        logger.error(traceback.format_exc())
    finally:
        from iondb.utils.files import rename_extension
        #undo edits
        rename_extension('etc/apt/', '.USBinstaller', '')
        if os.path.isfile('/etc/apt/sources.list.d/usb.list'):
            os.remove('/etc/apt/sources.list.d/usb.list')

    # update status
    from django.db import connection
    connection.close() # refresh dbase connection

    if success:
        models.GlobalConfig.objects.update(ts_update_status="Finished installing")
        models.Message.success("Upgrade completed successfully!")
    else:
        models.GlobalConfig.objects.update(ts_update_status="Install failure")
        models.Message.error("Upgrade failed during installation.")
    models.Message.objects.filter(expires="system-update-finished").delete()
    models.Message.objects.filter(tags__contains="new-upgrade").delete()


@app.task
def install_updates():
    logging.shutdown()
    try:
        run_as_daemon(_do_the_install)
    except Exception as err:
        logger.error("The daemonization of the TSconfig installer failed: %s" % err)
        from iondb.rundb import models
        models.GlobalConfig.objects.update(ts_update_status="Install failure")
        raise



# This can get stuck when NFS filesystems are misbehaving so need a timeout
@app.task(queue="diskutil", soft_time_limit=60)
def update_diskusage(fs_name):
    from iondb.utils.files import percent_full
    from iondb.rundb import models
    fs = models.FileServer.objects.get(name=fs_name)
    if os.path.exists(fs.filesPrefix):
        try:
            fs.percentfull = percent_full(fs.filesPrefix)
            fs.save()
            #logger.debug("Used space: %s %0.2f%%" % (fs.filesPrefix,fs.percentfull))
        except Exception as e:
            logger.warning ("could not update size of %s" % fs.filesPrefix)
            logger.error(e)
    else:
        logger.warning("directory does not exist on filesystem: %s" % fs.filesPrefix)
        fs.percentfull = 0
        fs.save()
    return fs_name


@task(queue='diskutil')
def post_update_diskusage(fs_name):
    '''Handler for update_diskusage task output'''
    from iondb.rundb import models

    inode_threshold = 0.90
    critical_threshold = 99
    warning_threshold = 95
    friendly_threshold = 70
    fs = models.FileServer.objects.get(name=fs_name)
    gc = models.GlobalConfig.get()
    dmfs = models.DMFileSet.objects.filter(version=settings.RELVERSION)

    #========================================================================
    # TS-6669: Banner Message when disk usage gets critical
    #========================================================================
    crit_tag = "%s_disk_usage_critical" % (fs.name)
    warn_tag = "%s_disk_usage_warning" % (fs.name)
    golink = "<a href='%s' >  Visit Data Management</a>" % ('/data/datamanagement/')
    if fs.percentfull > critical_threshold:
        msg = "* * * CRITICAL! %s: Partition is getting very full - %0.2f%% * * *" % (fs.filesPrefix, fs.percentfull)
        logger.debug(msg+"   %s" % golink)
        message = models.Message.objects.filter(tags__contains=crit_tag)
        if not message:
            models.Message.error(msg+"   %s" % golink, tags=crit_tag)
            notify_diskfull(msg)
    elif fs.percentfull > warning_threshold:
        msg = "%s: Partition is getting full - %0.2f%%" % (fs.filesPrefix, fs.percentfull)
        logger.debug(msg+"   %s" % golink)
        message = models.Message.objects.filter(tags__contains=warn_tag)
        if not message:
            models.Message.error(msg+"   %s" % golink, tags=warn_tag)
            notify_diskfull(msg)
    else:
        # Remove any message objects
        models.Message.objects.filter(tags__contains=crit_tag).delete()
        models.Message.objects.filter(tags__contains=warn_tag).delete()

    #========================================================================
    # Banner Message when Disk Management is not enabled
    #========================================================================
    try:
        friendly_tag = "%s_dm_not_enabled" % (fs.name)
        if not gc.auto_archive_enable and fs.percentfull > friendly_threshold:
            msg = "Data Management Auto Actions are not enabled and %s is %0.2f%% full" % (fs.filesPrefix, fs.percentfull)
            logger.debug(msg+"   %s" % golink)
            message = models.Message.objects.filter(tags__contains=friendly_tag)
            if not message:
                models.Message.error(msg+"   %s" % golink, tags=friendly_tag, route=models.Message.USER_STAFF)
                notify_diskfull(msg)
        else:
            models.Message.objects.filter(tags__contains=friendly_tag).delete()
    except:
        logger.error(traceback.format_exc())

    #========================================================================
    # Banner Message when Disk Management is enabled, but all auto-actions are disabled
    #========================================================================
    try:
        friendly_tag = "%s_all_auto_actions_disabled" % (fs.name)
        if gc.auto_archive_enable and dmfs.filter(auto_action="OFF").count() == 4:
            msg = "All Data Management Auto Actions are disabled and %s is %0.2f%% full" % (fs.filesPrefix, fs.percentfull)
            logger.debug(msg+"   %s" % golink)
            message = models.Message.objects.filter(tags__contains=friendly_tag)
            if not message:
                models.Message.error(msg+"   %s" % golink, tags=friendly_tag, route=models.Message.USER_STAFF)
                notify_diskfull(msg)
        else:
            models.Message.objects.filter(tags__contains=friendly_tag).delete()
    except:
        logger.error(traceback.format_exc())

    #========================================================================
    # Banner Message when inodes are running low
    #========================================================================
    try:
        inode_tag = "%s_low_inodes" % (fs.name)
        (itot, iuse, ifree) = file_utils.get_inodes(fs.filesPrefix)
        if float(iuse)/float(itot) > inode_threshold:
            msg = "Running out of inodes on %s. Used %d of %d. Contact IT or system support for further investigation." % (fs.filesPrefix, iuse, itot)
            logger.debug(msg+"   %s" % golink)
            message = models.Message.objects.filter(tags__contains=inode_tag)
            if not message:
                models.Message.error(msg, tags=inode_tag, route=models.Message.USER_STAFF)
                notify_diskfull(msg)
        else:
            models.Message.objects.filter(tags__contains=inode_tag).delete()
    except:
        logger.error(traceback.format_exc())


# Expires after 5 minutes; is scheduled every 10 minutes
# To trigger celery task from command line:
# python -c 'import iondb.bin.djangoinit, iondb.rundb.tasks as tasks; tasks.check_disk_space.apply_async()'
@periodic_task(run_every=600, expires=300, queue="periodic", ignore_result=True)
def check_disk_space():
    """
    For every FileServer object, start a task to get percentage of used disk space.
    Checks root partition for sufficient space.
    """
    from iondb.rundb import models
    from iondb.utils import files

    try:
        fileservers = models.FileServer.objects.all()
    except:
        logger.error(traceback.print_exc())
        fileservers = []

    for fs in fileservers:
        if os.path.exists(fs.filesPrefix):
            async_result = update_diskusage.apply_async([fs.name], link=post_update_diskusage.s())
        else:
            # log the error
            logger.warn("File Server does not exist.  Name: %s Path:%s" % (fs.name, fs.filesPrefix))

#        raidinfo = async_result.get(timeout=60)
#        fail_tag = "Failed getting disk usage for %s" % (fs.filesPrefix)
#        if async_result.failed():
#            logger.debug("%s" %(fail_tag))
#            message = models.Message.objects.filter(tags__contains=fail_tag)
#            if not message:
#                models.Message.error("%s at %s" % (fail_tag, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),tags=fail_tag)
#            continue
#        else:
#            models.Message.objects.filter(tags__contains=fail_tag).delete()

    #========================================================================
    # Check root partition free space
    #========================================================================
    root_threshold = 1048576 # 1 GB in KB
    try:
        warn_tag = "root_partition_space_warning"
        root_free_space = files.getSpaceKB('/')
        if root_free_space < root_threshold:
            msg = "Root Partition is getting full - less than 1GB available"
            logger.warn(msg)
            message = models.Message.objects.filter(tags__contains=warn_tag)
            if not message:
                models.Message.warn(msg, tags=warn_tag)
        else:
            models.Message.objects.filter(tags__contains=warn_tag).delete()
            #logger.debug("Root partition is spacious enough with %0.2f" % (root_free_space))
    except:
        logger.error(traceback.format_exc())


def notify_services_error(subject_line, msg, html_msg):
    '''sends an email with message'''
    logid = {'logid': "%s" % ('notify_services_error')}
    try:
        recipient = User.objects.get(username='it_contact').email
        logger.warning("it_contact is %s." % recipient, extra=logid)
    except:
        logger.warning("Could not retrieve it_contact.  No email sent.", extra=logid)
        return False
    # Check for blank email
    # TODO: check for valid email address
    if not recipient:
        logger.warning("No it_contact email configured.", extra=logid)
        return False

    #Needed to send email
    settings.EMAIL_HOST = 'localhost'
    settings.EMAIL_PORT = 25
    settings.EMAIL_USE_TLS = False

    try:
        site_name = models.GlobalConfig.get().site_name
    except:
        site_name = "Torrent Server"

    hname = socket.getfqdn()

    reply_to = 'donotreply@iontorrent.com'
    message = 'From: %s (%s)\n' % (site_name, hname)
    message += '\n'
    message += msg
    message += "\n"
    html_content = 'From: %s (<a href=%s>%s</a>)<br>' % (site_name, hname, hname)
    html_content += '<br>'
    html_content += html_msg
    html_content += "<br>"

    # Send the email
    try:
        recipient = recipient.replace(',', ' ').replace(';', ' ').split()
        logger.debug(recipient, extra=logid)
        sendthis = mail.EmailMultiAlternatives(subject_line, message, reply_to, recipient)
        sendthis.attach_alternative(html_content, "text/html")
        sendthis.send()
    except:
        logger.warning(traceback.format_exc(), extra=logid)
        return False
    else:
        logger.info("email sent for services alert", extra=logid)
        return True


def notify_diskfull(msg):
    '''sends an email with message'''
    logid = {'logid': "%s" % ('notify_diskfull')}
    try:
        recipient = User.objects.get(username='dm_contact').email
        logger.warning("dm_contact is %s." % recipient, extra=logid)
    except:
        logger.warning("Could not retrieve dm_contact.  No email sent.", extra=logid)
        return False

    # Check for blank email
    # TODO: check for valid email address
    if not recipient:
        logger.warning("No dm_contact email configured.  No email sent.", extra=logid)
        return False

    #Needed to send email
    settings.EMAIL_HOST = 'localhost'
    settings.EMAIL_PORT = 25
    settings.EMAIL_USE_TLS = False

    try:
        site_name = models.GlobalConfig.get().site_name
    except:
        site_name = "Torrent Server"

    hname = socket.getfqdn()

    subject_line = 'Torrent Server Data Management Disk Alert'
    reply_to = 'donotreply@iontorrent.com'
    message = 'From: %s (%s)\n' % (site_name, hname)
    message += '\n'
    message += msg
    message += "\n"

    # Send the email
    try:
        recipient = recipient.replace(',', ' ').replace(';', ' ').split()
        logger.debug(recipient, extra=logid)
        mail.send_mail(subject_line, message, reply_to, recipient)
    except:
        logger.warning(traceback.format_exc(), extra=logid)
        return False
    else:
        logger.info("Notification email sent for user acknowledgement", extra=logid)
        return True


#TS-5495: Refactored from a ionadmin cron job
#Note on time: we need to specify the time to run in UTC. 6am localtime
if time.localtime().tm_isdst and time.daylight:
    cronjobtime = 6 + int(time.altzone/60/60)
else:
    cronjobtime = 6 + int(time.timezone/60/60)
cronjobtime = (24 + cronjobtime) % 24


@periodic_task(run_every=crontab(hour=str(cronjobtime), minute="0", day_of_week="*"), queue="periodic")
def runnightly():
    import traceback
    from iondb.bin import nightly
    from iondb.rundb import models

    try:
        send_nightly = models.GlobalConfig.get().enable_nightly_email
        if send_nightly:
            nightly.send_nightly()
    except:
        logger.exception(traceback.format_exc())

    return


@app.task
def download_something(url, download_monitor_pk=None, dir="/tmp/", name="", auth=None):
    from iondb.rundb import models
    logger.debug("Downloading " + url)
    monitor, created = models.FileMonitor.objects.get_or_create(id=download_monitor_pk)
    monitor.url = url
    monitor.status = "Starting"
    monitor.celery_task_id = download_something.request.id
    monitor.name = name
    monitor.save()

    # create temp folder in dir to hold the thing
    local_dir = tempfile.mkdtemp(dir=dir)
    monitor.local_dir = local_dir
    monitor.save()
    os.chmod(local_dir, 0777)

    # open connection
    try:
        if auth:
            username, password = auth
            password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, url, username, password)
            handler = urllib2.HTTPBasicAuthHandler(password_mgr)
            opener = urllib2.build_opener(handler)
            resource = opener.open(url, timeout=30)
        else:
            resource = urllib2.urlopen(url, timeout=30)
    except urllib2.HTTPError as err:
        logger.error("download_something had an http error: %s" % err)
        monitor.status = "HTTP Error: {0}".format(err.msg)
        monitor.save()
        return None, monitor.id
    except urllib2.URLError as err:
        logger.error("download_something had a connection error: %s" % err)
        monitor.status = "Connection Error"
        monitor.save()
        return None, monitor.id
    except ValueError as err:
        logger.error("download_something got invalid url: " + str(url))
        monitor.status = "Invalid URL Error"
        monitor.save()
        return None, monitor.id

    #set the download's local file name if it wasn't specified above
    logger.info(resource.headers)
    if not name:
        # Check the server's suggested name
        pattern = re.compile(r"filename\s*=(.+)")
        match = pattern.search(resource.headers.get("Content-Disposition", ""))
        if match:
            name = match.group(1).strip()
    # name it something, for the love of humanity name it something!
    name = name or os.path.basename(urllib.unquote(urlparse.urlsplit(url)[2])) or 'the_download_with_no_name'
    monitor.name = name
    size = resource.headers.get("Content-Length", None)
    monitor.size = size and int(size)
    monitor.save()

    full_path = os.path.join(local_dir, name)
    # if not None, log the progress to the FileMonitor objects
    if monitor:
        monitor.status = "Downloading"
        monitor.save()

    # download and write CHUNK size bits of the file to disk at a time
    CHUNK = 16 * 1024
    tick = time.time()
    progress = 0
    try:
        with open(full_path, 'wb') as out_file:
            chunk = resource.read(CHUNK)
            while chunk:
                out_file.write(chunk)
                # accumulate the download progress as it happens
                progress += len(chunk)
                # every 2 seconds, save the current progress to the DB
                tock = time.time()
                if tock - tick >= 2:
                    monitor.progress += progress
                    monitor.save()
                    progress = 0
                    tick = tock
                chunk = resource.read(CHUNK)
        os.chmod(full_path, 0666)
    except IOError as err:
        monitor.status = "Connection Lost"
        monitor.save()
        return None, monitor.id

    monitor.progress += progress
    monitor.size = monitor.size or monitor.progress
    monitor.status = "Complete"
    monitor.save()

    return full_path, monitor.id
    # setup celery callback for success to take path and go
    # write generic err-back to log failure in DownloadProgress
    # write wrapper function to set up the common case.


@app.task
def ampliseq_zip_upload(args, meta):
    from iondb.rundb import publishers
    from iondb.rundb import models
    pub = models.Publisher.objects.get(name="BED")
    full_path, monitor_id = args
    monitor = models.FileMonitor.objects.get(id=monitor_id)
    upload = publishers.move_upload(pub, full_path, monitor.name, meta)
    publishers.run_pub_scripts(pub, upload)


@app.task(queue="slowlane")
def install_reference(args, reference_id, reference_mask_filename=None):
    from iondb.rundb import models

    def _is_genome_file(reference_path):
        ext = os.path.splitext(reference_path)[1]
        return ext.lower() in ['.fasta', '.fas', '.fa', '.fna', '.seq']

    reference = models.ReferenceGenome.objects.get(id=reference_id)
    reference.status = "preprocessing"
    reference.save()

    full_path, monitor_id = args
    if monitor_id:
        monitor = models.FileMonitor.objects.get(id=monitor_id)
        if monitor.status == "Complete":
            local_dir = monitor.local_dir
        else:
            reference.status = "download failed"
            reference.save()
            return
    else:
        local_dir = tempfile.mkdtemp(suffix=reference.short_name, dir=settings.TEMP_PATH)
        os.chmod(local_dir, 0777)
        shutil.move(reference.source, local_dir)
        full_path = os.path.join(local_dir, os.path.basename(reference.source))

    reference_mask_path = None
    logger.debug('Install reference %s (%d) from file: %s' % (reference.short_name, reference.pk, full_path))

    # handle gzip files
    if os.path.splitext(full_path)[1] == '.gz':
        success, gunzipped = check_gunzip(full_path)
        if success:
            full_path = gunzipped

    # handle ZIP files
    if zipfile.is_zipfile(full_path):
        extracted_path = os.path.join(local_dir, "reference_contents")
        extract_zip(full_path, extracted_path)

        reference.reference_path = extracted_path
        fasta_path = os.path.join(reference.reference_path, reference.short_name + '.fasta')

        # uploaded reference file name can be other than short_name and extension other than .fasta
        if not os.path.exists(fasta_path):
            files = filter(lambda x: _is_genome_file(x), os.listdir(reference.reference_path))
            if len(files) == 1:
                fasta_path = os.path.join(reference.reference_path, files[0])
            else:
                reference.status = "error"
                reference.verbose_error = "Error: upload must contain exactly one reference file"
                reference.save()
                logger.error('Error: Reference upload for %s (%d) had %d genome files' % (reference.short_name, reference.pk, len(files)))
                delete_that_folder(local_dir, "deleting uploaded reference temp files")
                return

        if reference_mask_filename:
            reference_mask_path = os.path.join(reference.reference_path, reference_mask_filename)
            #error if the specified reference_mask is not found
            if not os.path.exists(reference_mask_path):
                reference.status = "error"
                reference.verbose_error = "Error: reference upload refers to a non-existing file"
                reference.save()
                logger.error('Error: Reference upload for %s (%d) refers to a non-existing %s mask file' % (reference.short_name, reference.pk, reference_mask_filename))
                delete_that_folder(local_dir, "deleting uploaded reference temp files")
                return

    else:
        reference.reference_path = local_dir
        fasta_path = full_path
    reference.save()

    # preloaded references don't need to be indexed. Always force re-indexing if reference mask is used
    if reference.index_version == settings.TMAP_VERSION and not reference_mask_filename:
        logger.info('Skipping build tmap index for %s(%d), already indexed %s' % (reference.short_name, reference.pk, reference.index_version))
        reference.status = "complete"
        reference.enable_genome()
        delete_that_folder(local_dir, "deleting uploaded reference temp files")
    else:
        delete_task = delete_that_folder.subtask((local_dir, "deleting uploaded reference temp files"), immutable=True)
        async_result = build_tmap_index.apply_async((reference.id, fasta_path, reference_mask_path), link=delete_task)
        reference.celery_task_id = async_result.task_id
        reference.save()


@app.task (queue="w1", soft_time_limit=60)
def get_raid_stats_json():
    '''Wrapper to bash script calling MEGAraid tool'''
    raid_stats = None
    try:
        raidCMD = ["/usr/bin/ion_raidinfo_json"]
        q = subprocess.Popen(raidCMD, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = q.communicate()
        if q.returncode == 0:
            raid_stats = stdout
    except SoftTimeLimitExceeded:
        logger.error("get_raid_stats_json timed out")
    return raid_stats


@task (queue="w1", ignore_result=True)
def post_check_raid_status(raidinfo):
    '''Handler for output from get_raid_stats_json task'''
    from iondb.rundb import models
    raid_status = raid_utils.get_raid_status(raidinfo)
    if len(raid_status) > 0:
        message = models.Message.objects.filter(tags="raid_alert")
        # show alert for primary internal storage RAID
        if raid_utils.ERROR in [r.get('status') for r in raid_status]:
            if not message:
                msg = 'WARNING: RAID storage disk error.'
                golink = "<a href='%s' >  Visit Services Tab  </a>" % ('/configure/services/')
                models.Message.warn(msg+"   %s" % golink, tags="raid_alert")

                # TS-9461.  Send an email alert in conjunction with displaying the banner message
                msg = "WARNING: RAID storage disk error.\n"
                msg = msg+"\nVisit the Services Tab for more information"
                html_content = "WARNING: RAID storage disk error.<br>"
                html_content = html_content+"Visit the <a href='%s/%s' >Services Tab</a> for more information." % (socket.getfqdn(), 'configure/services/')
                notify_services_error('Torrent Server RAID Array Alert', msg, html_content)

        else:
            message.delete()


@periodic_task(run_every=timedelta(minutes=20), expires=600, queue="periodic", ignore_result=True)
def check_raid_status():
    '''check RAID state and alert user with message banner of any problems'''
    get_raid_stats_json.apply_async(link=post_check_raid_status.s())


@periodic_task(run_every=timedelta(hours=4), queue="periodic")
def update_news_posts():
    from iondb.rundb import models
    if not models.GlobalConfig.get().check_news_posts:
        return

    response = requests.get(settings.NEWS_FEED)
    if response.ok:
        feed = feedparser.parse(response.content)
        for article in feed['entries']:
            post_defaults = {
                "updated": dateutil.parser.parse(article['updated']),
                "summary": article.get('summary', ''),
                "title": article.get('title', 'Untitled'),
                "link": article.get('link', '')
            }
            # The news feed has been updated to include html content sections
            if len(article.get('content', [])) > 0:
                post_defaults["summary"] = article['content'][0].get('value', '')
            post, created = models.NewsPost.objects.get_or_create(guid=article.get('id', None), defaults=post_defaults)
        now = timezone.now()
        one_month = timedelta(days=30)
        # Delete all posts after #15 which are at least a month old.
        # i.e. all posts are kept for at least one month, and the newest 15 are always kept.
        for article in models.NewsPost.objects.order_by('-updated')[15:]:
            if now - article.updated > one_month:
                article.delete()
    else:
        logger.error("Could not get Ion Torrent news feed.")


@periodic_task(run_every=timedelta(minutes=3600), expires=600, queue="periodic")
def checkLspciForGpu():
    errorFileName = "/var/spool/ion/gpuErrors"
    gpuFound = False
    revsValid = True
    count = 0

    # get the output of lspci

    # requires Python 2.7
    #lspciStr = subprocess.check_output("lspci")

    # works with Python 2.6.5
    p = subprocess.Popen(['lspci'], stdout=subprocess.PIPE)
    lspciStr, err = p.communicate()

    # find all the lines containing "nvidia" (case insensitive) and get the rev
    startIndex = 0;
    while True:
        revNum, startIndex = findNvidiaInLspci(lspciStr, startIndex)
        #print "revNum", revNum, "startIndex" , startIndex

        # if we didn't find a line containing nvidia, bail
        if (startIndex == -1):
            break

        gpuFound = True

        # check the rev num
        if revNum == 'ff':     # When rev == ff, we have lost GPU connection
            revsValid = False

        # sanity check
        count = count + 1
        if count > 32:
            break

    writeError(errorFileName, gpuFound, revsValid)

    return gpuFound and revsValid


def findNvidiaInLspci(lspciStr, startIndex):

    # find the line with the NVIDIA controller information
    lowStr = lspciStr.lower()
    idx = lowStr.find("controller: nvidia", startIndex)

    # if we didn't find it, bail
    if (idx == -1):
        return "", -1

    # truncate the line with the NVIDIA info
    newline = lspciStr.find("\n", idx)
    if newline != -1:
        nvidiaLine = lspciStr[idx:newline]

    # extract the rev number from the NVIDIA line
    token = "(rev "
    beg = nvidiaLine.find(token) + len(token)
    end = nvidiaLine.find(")", beg)

    return nvidiaLine[beg:end], newline + 1


def writeError(errorFileName, gpuFound, allRevsValid):
    with open(errorFileName, 'w') as f:
        f.write(json.dumps({'gpuFound': gpuFound, 'allRevsValid': allRevsValid}))


@app.task
def lock_ion_apt_sources(enable=False):
    """Set sources.list to point to Ion archive location for current version
    Change from:
        deb http://ionupdates.com/updates/software lucid/
        to:
        deb http://ionupdates.com/updates/software/archive 4.0.2/
    -or-
        vicey-versy
    """
    process = subprocess.Popen(['sudo', '/opt/ion/iondb/bin/lock_ion_apt_sources.py', str(enable)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    logger.info(output)
    logger.error(err)


@periodic_task(run_every=timedelta(minutes=10), expires=300, queue="periodic")
def post_run_nodetests():
    '''
        Alert user with message banner and email of any problems with cluster node
        Autodisables failed nodes if GlobalConfig.cluster_auto_disable flag is set
        Also updates queue info for the cluster
    '''
    from iondb.rundb.models import Message, Cruncher, GlobalConfig

    bad_nodes = Cruncher.objects.exclude(state='G')
    message = Message.objects.filter(tags="cluster_alert")
    if len(bad_nodes) > 0:
        auto_disable = GlobalConfig.get().cluster_auto_disable
        if not message:
            msg = 'WARNING: Cluster node failure.'
            golink = "<a href='%s' >  Visit Services Tab  </a>" % ('/configure/services/')
            Message.warn(msg+"   %s" % golink, tags="cluster_alert")

            # TS-9461. Send an email in conjunction with displaying the banner message
            msg = "The following nodes have been disabled due to an error condition.\n"
            for node in bad_nodes:
                msg = msg+node.name+"\n"
            msg = msg+"\nVisit the Services Tab for more information"
            msg = msg+"http://%s/configure/services" % (socket.getfqdn())
            if auto_disable:
                html_content = "The following nodes have been disabled due to an error condition.<br>"
            else:
                html_content = "The following nodes have an error condition.<br>"
            html_content += "<ul>"
            for node in bad_nodes:
                err_status = node.info['error']
                html_content = html_content+"<li>"+node.name+" - "+err_status+"</li>"
            html_content += "</ul>"
            html_content = html_content+"Visit the <a href='%s/%s' >Services Tab</a> for more information." % (socket.getfqdn(), 'configure/services/')
            notify_services_error('Torrent Server Cluster Node Alert', msg, html_content)

        # disable nodes that fail nfs_mount_test or version_test
        if auto_disable:
            disable_on_tests = ['nfs_mount_test', 'version_test']
            for node in bad_nodes:
                queues = node.info.get('queues')
                node_disabled = queues['disabled'] == queues['total'] if queues else False
                if not node_disabled and 'config_tests' in node.info:
                    failed_tests = sum(test[1] == 'failure' for test in node.info['config_tests'] if test[0] in disable_on_tests)
                    if failed_tests > 0:
                        logger.info('Node %s failed tests and will be disabled' % node.name)
                        cluster_ctrl_task("disable", node.name, "system")
    else:
        message.delete()


@task(queue='w1', soft_time_limit=60, expires=600)
def test_node_and_update_db(node, head_versions):
    """
    Test connectivity to a cluster node
    :parameter node: Host name of the cluster node to probe
    :parameter head_versions: a dictionary of version of packages on the head node
    """

    logger.info("Testing node: %s" % node.name)
    node_status = {
        'name': node.name,
        'status': '',
        'connect_tests': [],
        'error': ''
    }
    try:
        node_status = connect_nodetest(node.name)
    except SoftTimeLimitExceeded:
        logger.error("Time limit exceeded for connect_nodetest on %s" % node.name)
        node_status['status'] = 'error'
        node_status['error'] = "Time limit exceeded for connect_nodetest()"
    except:
        logger.error(traceback.format_exc())

    if node_status['status'] == 'good':
        logger.info("Node: %s passed connect test" % node.name)
        try:
            node_status.update(config_nodetest(node.name, head_versions))
            logger.info("Node: %s passed config test" % node.name)
        except SoftTimeLimitExceeded:
            logger.error("Time limit exceeded for config_nodetest on %s" % node.name)
            node_status['status'] = 'error'
            node_status['error'] = "Time limit exceeded for config_nodetest().  NFS mounts could be hung up."
        except:
            logger.error(traceback.format_exc())

    try:
        add_eventlog(node, node_status)
    except:
        # TODO
        pass

    # update queue state
    node_status["queues"] = queue_info(node.name)

    # update cruncher database entry
    node.state = node_status['status'][0].upper()
    node.info = node_status
    node.save()

    return node.name, node_status['status']


@periodic_task(run_every=timedelta(minutes=20), expires=600, queue="periodic", ignore_result=True)
def check_cluster_status():
    """
    Runs a periodic test for all cluster nodes.
    """
    # run tests for cluster nodes
    from ion.utils.TSversion import findVersions
    from iondb.rundb.models import Cruncher

    nodes = Cruncher.objects.all().order_by('pk')
    if nodes:
        # launch parallel celery tasks to test all nodes
        # Note that parallelism will be limited by the number of workers in a queue.
        head_versions, _ = findVersions()
        result = group(test_node_and_update_db.s(node, head_versions) for node in nodes).apply_async()
        return result


@app.task(queue='w1')
def cluster_ctrl_task(action, name, username):
    ''' send SGE commands, run as task to get root permissions '''
    from iondb.rundb.models import Cruncher, EventLog

    nodes = Cruncher.objects.filter(name=name) if name != "all" else Cruncher.objects.all()
    if not nodes:
        return "Node %s not found" % name

    errors = []
    info = queue_info()
    for node in nodes:
        # check if already in desired state
        queues = info.get(node.name)

        if action == "enable" and queues['disabled'] == 0:
            error = 'SGE queues for %s are already enabled' % node.name
        elif action == "disable" and queues['disabled'] == queues['total']:
            error = 'SGE queues for %s are already disabled' % node.name
        else:
            error = sge_ctrl(action, node.name)

        if not error:
            msg = "%s SGE queues" % action.capitalize()
            EventLog.objects.add_entry(node, msg, username)
        else:
            errors.append(error)

    # update queue info after changing state
    info = queue_info()
    for node in nodes:
        node.info["queues"] = info.get(node.name)
        node.save()

    return errors


def add_eventlog(node, new_status):
    '''
    # add eventlog on following conditions:
    #    node is new: no log entries exist
    #    node state or error messages changed
    #    any node test returns different status

    # NOTE: ClusterInfoHistoryResource parses log messages for History page, if changing format you must update api.py
    '''
    from iondb.rundb.models import EventLog
    addlog = EventLog.objects.filter(object_pk=node.pk).count() == 0

    if str(node.state) != str(new_status['status'][0].upper()):
        msg = "%s state changed from %s to %s<br>" % (node.name, node.get_state_display().title(), new_status['status'].title())
        addlog = True
    else:
        msg = "%s state is %s<br>" % (node.name, new_status['status'].title())

    addlog = addlog or node.info.get('error', '') != new_status.get('error', '')
    addlog = addlog or node.info.get('version_test_errors', '') != new_status.get('version_test_errors', '')
    msg += "Error: %s %s<br>" % (new_status.get('error', ''), new_status.get('version_test_errors', ''))

    old_test_results = dict(node.info.get('config_tests', []) + node.info.get('connect_tests', []))
    for test_name, test_result in new_status.get('config_tests', []) + new_status.get('connect_tests', []):
        msg += "%s: %s<br>" % (test_name, test_result)
        addlog = addlog or old_test_results.get(test_name, '') != test_result

    if addlog:
        EventLog.objects.add_entry(node, msg, username="system")


def generate_TF_files(tfkey, tf_dir='/results/referenceLibrary/TestFragment'):
    # Creates DefaultTFs.conf, and fasta and index files per TF key
    from iondb.rundb.models import Template

    tfconf_path = os.path.join(tf_dir, "DefaultTFs.conf")
    fasta_filename = "DefaultTFs.fasta"
    dest = os.path.join(tf_dir, tfkey)

    tfs = Template.objects.filter(isofficial=True).order_by('name')
    os.umask(0000)

    # write conf file
    lines = ["%s,%s,%s" % (tf.name, tf.key, tf.sequence,) for tf in tfs]
    with open(tfconf_path, 'w') as f:
        f.write('\n'.join(lines))

    tfs = tfs.filter(key=tfkey)
    if len(tfs) > 0:
        if not os.path.exists(dest):
            os.mkdir(dest)

        # write fasta file
        fasta_path = os.path.join(dest, fasta_filename)
        with open(fasta_path, 'w') as f:
            for tf in tfs:
                f.write('>%s\n' % tf.name)
                f.write('%s\n' % tf.sequence.strip())

        # make faidx index file
        subprocess.check_call("/usr/local/bin/samtools faidx %s" % fasta_path, shell=True)
        # make tmap index files
        subprocess.check_call("/usr/local/bin/tmap index -f %s" % fasta_path, shell=True)
    else:
        # remove key folder if this key no longer has any TFs
        try:
            shutil.rmtree(dest)
        except OSError:
            logger.error("Failed to delete folder %s" % dest)


def check_gunzip(gunZipFile, logger=None):
    #import mimetypes
    isTaskSuccess = False
    if not logger:
        logger = logging.getLogger(__name__)
    #Extract if annotation file is in gzip format
    try:
        result = subprocess.check_call("gunzip %s" % gunZipFile, shell=True)
        if not result: #extract failed If non-zero exit status
            isTaskSuccess = True
            fileToRegister = re.sub('.gz$', '', gunZipFile)
            return isTaskSuccess, fileToRegister
    except Exception, err:
        logger.error("Failed to extract .gz file %s" % err)

    return isTaskSuccess, gunZipFile


@app.task
def new_annotation_download(annot_url, updateVersion, **reference_args):
    ref_short_Name = reference_args['short_name']
    from iondb.rundb.models import ReferenceGenome, Publisher
    from django.core.files import File
    from iondb.rundb import publishers
    fileToRegister = None
    isTaskSuccess = False
    try:
        reference = ReferenceGenome.objects.get(short_name=ref_short_Name)
    except Exception, err:
        logger.debug("Reference does not exists for  Annotation File {0} with version {1}".format(annot_url, updateVersion))
        return err
    try:
        (isTaskSuccess, fileToRegister, downloadstatus) = start_annotation_download(annot_url, reference, updateVersion=updateVersion)
        print (isTaskSuccess, fileToRegister, downloadstatus)
    except Exception as Err:
        logger.debug("System Error {0}".format(Err))

    if isTaskSuccess and downloadstatus == "Complete":
        #convert the raw file into Django File object so that publisher framework can accept it
        fileObject = open(fileToRegister)
        upload = File(fileObject)
        file_name = os.path.basename(upload.name)
        upload.name = file_name
        #Go ahead and register the annotation file via publisher framework
        pub_name = "refAnnot"
        meta = {"publisher": pub_name, "reference": ref_short_Name, "annotation_url": annot_url, "upload_type": "Annotation"}
        try:
            pub = Publisher.objects.get(name=pub_name)
            contentUpload, _ = publishers.edit_upload(pub, upload, json.dumps(meta))
            return contentUpload
        except:
            logger.debug("Publisher does not exists {0}".format(pub_name))

    return isTaskSuccess


def start_annotation_download(annot_url, reference, callback=None, updateVersion=None, monitor=None):
    from iondb.rundb.models import FileMonitor
    import mimetypes
    tagsInfo = "reference_annotation_{0}".format(float(updateVersion))
    monitor = FileMonitor(url=annot_url, tags=tagsInfo)
    monitor.status = "Downloading"
    monitor.save()
    fileToRegister = None
    downloaded_fileTempPath = None
    #logger.error("File Not Found (404) URL Info: %s" % (annot_url))
    try:
        urllib2.urlopen(annot_url)
    except Exception as err:
        monitor.status = "System Error" + str(err)
        monitor.save()
        logger.error("File Not Found (404) URL Info: %s %s" % (annot_url, str(err)))
        return (False, fileToRegister, monitor.status)
    try:
        download_args = (annot_url, monitor.id, settings.TEMP_PATH)
        async_result = download_something.apply_async(download_args, refID=reference.id)
        logger.error(async_result)

        isTaskSuccess = False
        if async_result.status == 'PENDING':
            try:
                async_result.get()
            except Exception as err:
                monitor.status = "Download Error"
                monitor.save()
                logger.error("Error in Download File '{0}': {1}".format(annot_url, err))
        if async_result.status == 'SUCCESS':
            monitor = FileMonitor.objects.get(tags=tagsInfo, url=annot_url)
            isTaskSuccess = async_result.successful()
            result = async_result.result
            downloaded_fileTempPath = result[0]
        else:
            monitor.status = "Downloading"
            monitor.save()

        if isTaskSuccess:
            fileToRegister = downloaded_fileTempPath

            gztype = mimetypes.guess_type(fileToRegister)
            if gztype[1] == 'gzip':
                (isExtractSuccess, fileToRegister) = check_gunzip(fileToRegister)
                if not isExtractSuccess:
                    monitor.status = ".gz Annotation file extraction Failed"
                    monitor.save()

        return (isTaskSuccess, fileToRegister, monitor.status)

    except Exception as err:
        logger.debug("System Error: Caused Unknown Exception {0}".format(err))
        monitor.status = "System Error. Please contact TS administrator"
        monitor.save()


@app.task(queue="w1")
def set_timezone(request):
    """
    Sets the timezone data
    """
    with open ('/etc/timezone', 'w') as f:
        f.write(request)
    reconfig = subprocess.Popen(['dpkg-reconfigure', '-f', "noninteractive", "tzdata"], stdout=subprocess.PIPE)
    reconfig.communicate()
    return reconfig.returncode


@app.task
def install_BED_files(bedfileList, callback=None):
    '''
        Launches a set of tasks to download and install multiple BED files
        Optionally adds a callback to run after all install tasks are complete
    '''
    from iondb.rundb.models import FileMonitor
    from iondb.rundb.publishers import publish_file

    logger.info('install_BED_files: received files to process %s' % ', '.join([b['source'] for b in bedfileList]))
    bedfile_tasks = []
    for info in bedfileList:
        monitor = FileMonitor(url=info['source'], tags="bedfile")
        monitor.save()

        publish_task = publish_file.s('BED', json.dumps(info))
        bedfile_tasks.append( download_something.subtask((monitor.url, monitor.id), link=publish_task) )

    if callback:
        async_result = chord( bedfile_tasks )(callback)
    else:
        async_result = group( bedfile_tasks )()
    return async_result


@app.task
def release_tasklock(lock_id, parent_lock_id=''):
    from iondb.utils.TaskLock import TaskLock
    from iondb.rundb.models import Message
    applock = TaskLock(lock_id)
    applock.unlock()
    logger.info("Worker PID %d lock_id unlocked %s" % (os.getpid(), lock_id))

    # if all children processes are done can release the parent lock
    if parent_lock_id:
        parent_lock = TaskLock(parent_lock_id)
        if all([ TaskLock(child_id).get() is None for child_id in parent_lock.get()]):
            parent_lock.unlock()
            logger.info("Worker PID %d parent lock_id unlocked %s" % (os.getpid(), parent_lock_id))
            Message.info("Reference and BED files installation complete for %s" % parent_lock_id, tags="install_"+parent_lock_id)
