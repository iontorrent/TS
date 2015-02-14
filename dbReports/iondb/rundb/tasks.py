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
from ion.utils.timeout import timeout
from iondb.utils import raid as raid_utils
from iondb.utils import files as file_utils

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
            raise OSError("Must extract zip file to a directory. File already exists: '%s'", dest)
        if dest.find(settings.PLUGIN_PATH) == 0:
            ## Only delete content under PLUGIN_PATH.
            delete_that_folder(dest, "Deleting content at destination path '%s'" % dest)
        else:
            raise OSError("Unable to extract ZIP - directory '%s' already exists", dest)
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
            if ((member.external_attr <<16L) & 0120000):
                logging.error("ZIP archive contains symlink: '%s'. Skipping.", member.filename)
                continue

            if "__MACOSX" in filename:
                logging.warn("ZIP archive contains __MACOSX meta folder. Skipping", member.filename)
                continue

            # Get permission set inside archive
            perm = ((member.external_attr >> 16L) & 0777 ) or 0755

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
                with os.fdopen(os.open(targetpath, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, perm),'wb') as targetfh:
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
                logger.warn("Failed to set time/owner attributes on '%s': %s", targetpath , e)

            extracted_files.append(targetpath)

    return (prefix, extracted_files)

def unzipPlugin(zipfile, logger=None):
    if not logger:
        logger = logging.getLogger(__name__)
    ## Extract plugin to scratch folder. When complete, move to final location.
    plugin_path, ext = os.path.splitext(zipfile)
    plugin_name = os.path.basename(plugin_path)

    # ZIP file must named with plugin name - fragile
    # FIXME - handle (1) additions (common for multiple downloads via browser)
    # FIXME - handle version string in ZIP archive name

    scratch_path = os.path.join(settings.PLUGIN_PATH,"scratch","install-temp",plugin_name)
    (prefix, files) = extract_zip(zipfile, scratch_path, auto_prefix=True, logger=logger)
    if prefix:
        plugin_name = os.path.basename(prefix)

    plugin_temp_home = os.path.join(scratch_path, prefix)
    try:
        # Convert script into PluginClass, get info by introspection
        from iondb.plugins.manager import pluginmanager
        script, islaunch = pluginmanager.find_pluginscript(plugin_temp_home, plugin_name)
        logger.debug("Got script: %s", script)
        from ion.plugin.loader import cache
        ret = cache.load_module(plugin_name, script)
        cls = cache.get_plugin(plugin_name)
        p = cls()
        final_name = p.name # what the plugin calls itself, regardless of ZIP file name
        logger.info("Plugin calls itself: '%s'", final_name)
    except:
        logger.exception("Unable to interrogate plugin name from: '%s'", zipfile)
        final_name = plugin_name

    #move to the plugin dir
    # New extract_zip removes prefix from extracted files.
    # But still writes to file_name
    try:
        final_install_dir =  os.path.join(settings.PLUGIN_PATH, final_name)
        if os.path.exists(final_install_dir) and (final_install_dir != settings.PLUGIN_PATH):
            logger.info("Deleting old copy of plugin at '%s'", final_install_dir)
            delete_that_folder(final_install_dir, "Error Deleting old copy of plugin at '%s'" % final_install_dir)
        parent_folder = os.path.dirname(final_install_dir)
        if not os.path.exists(parent_folder):
            logger.info("Creating path for plugin '%s' for '%s'", parent_folder, final_install_dir)
            os.makedirs(parent_folder, 0555)

        logger.info("Moving plugin from temp extract folder '%s' to final location: '%s'", plugin_temp_home, final_install_dir)
        shutil.move(plugin_temp_home, final_install_dir)
        delete_that_folder(scratch_path, "Deleting plugin install scratch folder")
    except (IOError, OSError):
        logger.exception("Failed to move plugin from temp extract folder '%s' to final location: '%s'", plugin_temp_home, final_install_dir)
        raise

    # Now that it has been downloaded,
    # convert pre-plugin into real db plugin object
    try:
        from iondb.plugins.manager import pluginmanager
        (new_plugin, updated) = pluginmanager.install(final_name, final_install_dir)
    except ValueError:
        logger.exception("Failed to install plugin")
        #delete_that_folder(final_install_dir)

    return {
        "plugin": final_name,
        "path": final_install_dir,
        "files": files,
    }


@app.task
def echo(message, wait=0.0):
    time.sleep(wait)
    logger.info("Logged: " + message)
    print(message)

@app.task
def delete_that_folder(directory, message):
    def delete_error(func, path, info):
        logger.error("Failed to delete %s: %s", path, message)

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
    temp_path_uniq = os.path.join(temp_path,uuid_path)
    os.mkdir(temp_path_uniq)

    try:
        file = os.path.join(temp_path_uniq,baseFile)

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
    except urllib2.HTTPError, e:
        logger.error("HTTP Error: %d '%s'",e.code , url)
        delete_that_folder(temp_path_uniq, "after download error")
        return False
    except urllib2.URLError, e:
        logger.error("URL Error: %s '%s'",e.reason , url)
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

from . import zeroinstallHelper

# Helper for downloadPlugin task
def downloadPluginZeroInstall(url, plugin, logger=None):
    """ To be called for zeroinstall xml feed urls.
        Returns plugin prototype, not full plugin model object.
    """
    try:
        downloaded  = zeroinstallHelper.downloadZeroFeed(url)
        feedName = zeroinstallHelper.getFeedName(url)
    except:
        logger.exception("Failed to fetch zeroinstall feed")
        plugin.status["installStatus"] = "failed"
        plugin.status["result"] = str(sys.exc_info()[1][0])
        return False

    # The url field stores the zeroinstall feed url
    plugin.url = url
    plugin.name = feedName.replace(" ","")

    if not downloaded:
        logger.error("Failed to download url: '%s'", url)
        plugin.status["installStatus"] = "failed"
        plugin.status["result"] = "processed"
        return False

    plugin.status["installStatus"] = "installed"

    # Find plugin in subdirectory of extracted and installed path
    for d in os.listdir(downloaded):
        # Skip MACOSX attribute zip artifact
        if d == '__MACOSX':
            continue
        nestedpath = os.path.join(downloaded, d)
        if not os.path.isdir(nestedpath):
            continue
        # only take subdirectory with launch.sh script
        if os.path.exists(os.path.join(nestedpath, 'launch.sh')):
            plugin.path = os.path.normpath(nestedpath)
            break
        if os.path.exists(os.path.join(nestedpath, plugin.name + '.py')):
            plugin.path = os.path.normpath(nestedpath)
            break
    else:
        # Plugin expanded without top level folder
        plugin.path = downloaded
        # assert launch.sh exists?

    plugin.status["result"] = "0install"
    # Other fields we can get from zeroinstall feed?

    logger.debug(plugin)
    # Version is parsed during install - from launch.sh, ignoring feed value
    return plugin

# Helper for downloadPlugin task
def downloadPluginArchive(url, plugin, logger=None):
    ret = downloadChunks(url)
    if not ret:
        plugin.status["installStatus"] = "failed"
        plugin.status["result"] = "failed to download '%s'" % url
        return False
    downloaded, url = ret

    pdata = unzipPlugin(downloaded, logger=logger)

    plugin.name = pdata['plugin'] or os.path.splitext(os.path.basename(url))[0]
    plugin.path = pdata['path'] or os.path.join(settings.PLUGIN_PATH, plugin.name )

    #clean up archive file and temp dir (archive should be only file in dir)
    os.unlink(downloaded)
    os.rmdir(os.path.dirname(downloaded))

    if unzipStatus:
        plugin.status["result"] = "unzipped"
    else:
        plugin.status["result"] = "failed to unzip"


    return True

@app.task
def downloadPlugin(url, plugin=None, zipFile=None):
    """download a plugin, extract and install it"""
    if not plugin:
        from iondb.rundb import models
        plugin = models.Plugin.objects.create(name='Unknown', version='Unknown', status={})
    plugin.status["installStatus"] = "downloading"

    #normalise the URL
    url = urlparse.urlsplit(url).geturl()

    if not zipFile:
        if url.endswith(".xml"):
            status = downloadPluginZeroInstall(url, plugin, logger=logger)
            logger.error("xml") # logfile
        else:
            status = downloadPluginArchive(url, plugin, logger=logger)
            logger.error("zip") # logfile

        if not status:
            # FIXME - Errors!
            installStatus = plugin.status.get('installStatus', 'Unknown')
            result = plugin.status.get('result', 'unknown')
            msg = "Plugin install '%s', Result: '%s'" % (installStatus, result)

            logger.error(msg) # logfile
            from iondb.rundb import models
            models.Message.error(msg) # UI message
            return False
    else:
        # Extract zipfile
        scratch_path = os.path.join(settings.PLUGIN_PATH,"scratch")
        zip_file = os.path.join(scratch_path, zipFile)
        plugin.status["installStatus"] = "extracting zip"

        try:
            ret = unzipPlugin(zip_file)
        except:
            logger.exception("Failed to unzip Plugin: '%s'", zip_file)
        finally:
            #remove the zip file
            os.unlink(zip_file)

        plugin.name = ret['plugin']
        plugin.path = ret['path']
        plugin.status["installStatus"] = "installing from zip"

    # Now that it has been downloaded,
    # convert pre-plugin into real db plugin object
    try:
        from iondb.plugins.manager import pluginmanager
        (new_plugin, updated) = pluginmanager.install(plugin.name, plugin.path)
    except ValueError:
        logger.exception("Plugin rejected by installer. Check syntax and content.")
        return None

    # Copy over download status messages and url
    new_plugin.status = plugin.status
    if plugin.url:
        new_plugin.url = plugin.url
    new_plugin.save()

    logger.info("Successfully downloaded and installed plugin %s v%s from '%s'", new_plugin.name, new_plugin.version, url)

    return new_plugin

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
def static_ip(address, subnet, gateway):
    """Usage: TSstaticip [options]
         --ip      Define host IP address
         --nm      Define subnet mask (netmask)
         --nw      Define network ID
         --bc      Define broadcast IP address
         --gw      Define gateway/router IP address
    """
    cmd = ["/usr/sbin/TSstaticip",
           "--ip", address,
           "--nm", subnet,
           "--gw", gateway,
           ]
    logger.info("Network: Setting host static, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@app.task
def dhcp():
    """Usage: TSstaticip [options]
        --remove  Sets up dhcp, removing any static IP settings
    """
    cmd = ["/usr/sbin/TSstaticip", "--remove"]
    logger.info("Network: Setting host DHCP, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@app.task
def proxyconf(address, port, username, password):
    """Usage: TSsetproxy [options]
         --address     Proxy address (http://proxy.net)
         --port         Proxy port number
         --username    Username for authentication
         --password    Password for authentication
         --remove      Removes proxy setting
    """
    cmd = ["/usr/sbin/TSsetproxy",
           "--address", address,
           "--port", port,
           "--username", username,
           "--password", password,
           ]
    logger.info("Network: Setting proxy settings, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@app.task
def ax_proxy():
    """Usage: TSsetproxy [options]
         --remove      Removes proxy setting
    """
    cmd = ["/usr/sbin/TSsetproxy", "--remove"]
    logger.info("Network: Removing proxy settings, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@app.task
def dnsconf(dns):
    """Usage: TSdns [options]
         --dns      Define one or more comma delimited dns servers
    """
    cmd = ["/usr/sbin/TSdns", "--dns", dns]
    logger.info("Network: Changing DNS settings, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
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
        otStatus = open("/tmp/OTstatus",'w').close()
        #run the onetouch update script
        try:
            updateStatus = findHosts.findOneTouches()
        except:
            updateStatus = "FAILED"
        otStatus = open("/tmp/OTstatus",'w')
        otStatus.write(str(updateStatus) + "\n")
        otStatus.write( "DONE\n")
        otStatus.close()
        #now remove the lock
        os.unlink("/tmp/OTlock")
        return True

    return False


def make_reference_paths(reference):
    reference.reference_path = os.path.join(settings.TMAP_DISABLED_DIR, reference.short_name)
    os.mkdir(reference.reference_path)
    reference.save()
    return os.path.join(reference.reference_path, reference.short_name + ".fasta")


@app.task(queue="slowlane")
def unzip_reference(reference_id, reference_file=None):
    from iondb.rundb import models
    reference = models.ReferenceGenome.objects.get(pk=reference_id)
    reference.status = "preprocessing"
    reference.save()
    zip_path = reference.file_monitor.full_path()
    destination = make_reference_paths(reference)
    try:
        archive =  zipfile.ZipFile(zip_path)
        source_file = archive.open(reference_file, 'rU')
        dest_file = open(destination, 'w')
        shutil.copyfileobj(source_file, dest_file)
        archive.close()
    except Exception as err:
        logger.error("Could extract fasta zipped reference id=%d at %s" % (reference.pk, zip_path))
        raise err
    reference.file_monitor.delete()
    return destination


@app.task(queue="slowlane")
def copy_reference(reference_id):
    from iondb.rundb import models
    reference = models.ReferenceGenome.objects.get(pk=reference_id)
    reference.status = "preprocessing"
    reference.save()
    fasta_path = reference.file_monitor.full_path()
    destination = make_reference_paths(reference)
    shutil.move(fasta_path, destination)
    reference.file_monitor.delete()
    return destination


@app.task(queue="slowlane")
def build_tmap_index(reference_id):
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
    fasta = os.path.join(reference.reference_path , reference.short_name + ".fasta")
    logger.debug("TMAP %s rebuild, for reference %s(%d) using fasta %s"%
         (settings.TMAP_VERSION, reference.short_name, reference.pk, fasta))

    cmd = [
        '/usr/local/bin/build_genome_index.pl',
        "--auto-fix",
        "--fasta", fasta,
        "--genome-name-short", reference.short_name,
        "--genome-name-long", reference.name,
        "--genome-version", reference.version
    ]

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
        reference.enabled = True
        reference.enable_genome()
    else:
        logger.error('TMAP index rebuild "%s" failed:\n%s' %
                     (" ".join(cmd), stderr))
        reference.status = 'error'
        reference.verbose_error = json.dumps((stdout, stderr, ret))
        reference.save()
        reference.enabled = False
        reference.disable_genome()

    return ret == 0


def IonReporterWorkflows(autorun=True):

    try:
        from iondb.rundb import models
        if autorun:
            IonReporterUploader= models.Plugin.objects.get(name="IonReporterUploader_V1_0",selected=True,active=True,autorun=True)
        else:
            IonReporterUploader= models.Plugin.objects.get(name="IonReporterUploader_V1_0",selected=True,active=True)

        logging.error(IonReporterUploader.config)
        config = IonReporterUploader.config
    except models.Plugin.DoesNotExist:
        error = "IonReporterUploader V1.0 Plugin Not Found."
        logging.error(error)
        return False, error

    try:
        headers = {"Authorization" : config["token"] }
        url = config["protocol"] + "://" + config["server"] + ":" + config["port"] +"/grws/analysis/wflist"
        logging.info(url)
    except KeyError:
        error = "IonReporterUploader V1.0 Plugin Config is missing needed data."
        logging.exception(error)
        return False, error

    try:
        #using urllib2 right now because it does NOT verify SSL certs
        req = urllib2.Request(url = url, headers = headers)
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
        IonReporterUploader= models.Plugin.objects.get(name=plugin,selected=True,active=True)
        logging.error(IonReporterUploader.config)
        config = IonReporterUploader.config
    except models.Plugin.DoesNotExist:
        error = plugin + " Plugin Not Found."
        logging.exception(error)
        return False, error

    try:
        headers = {"Authorization" : config["token"] }
        url = config["protocol"] + "://" + config["server"] + ":" + config["port"] + "/grws_1_2/data/versionList"
        logging.info(url)
    except KeyError:
        error = plugin + " Plugin Config is missing needed data."
        logging.debug(plugin +" config: " + config)
        logging.exception(error)
        return False, error

    try:
        #using urllib2 right now because it does NOT verify SSL certs
        req = urllib2.Request(url = url, headers = headers)
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
def scheduled_update_check():
    from iondb.rundb import models
    try:
        packages = check_updates()
        upgrade_message = models.Message.objects.filter(tags__contains="new-upgrade")
        if packages:
            if not upgrade_message.all():
                models.Message.info('There is an update available for your Torrent Server. <a class="btn btn-success" href="/admin/update">Update Now</a>', tags='new-upgrade', route='_StaffOnly')
            download_now = models.GlobalConfig.get().enable_auto_pkg_dl
            if download_now:
                async = download_updates.delay()
                logger.debug("Auto starting download of %d packages in task %s" % (len(packages), async.task_id))
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
    try:
        import ion_tsconfig.TSconfig
        tsconfig = ion_tsconfig.TSconfig.TSconfig()
        enable_security_update = models.GlobalConfig.get().enable_auto_security
        tsconfig.set_securityinstall(enable_security_update)
        packages = tsconfig.TSpoll_pkgs()
    except Exception as err:
        logger.error("TSConfig raised '%s' during update check." % err)
        models.GlobalConfig.objects.update(ts_update_status="Update failure")
        raise

    return packages

@app.task
def download_updates(auto_install=False):
    from iondb.rundb import models
    try:
        import ion_tsconfig.TSconfig
        tsconfig = ion_tsconfig.TSconfig.TSconfig()
        enable_security_update = models.GlobalConfig.get().enable_auto_security
        tsconfig.set_securityinstall(enable_security_update)
        downloaded = tsconfig.TSexec_download()
    except Exception as err:
        logger.error("TSConfig raised '%s' during a download" % err)
        models.GlobalConfig.objects.update(ts_update_status="Download failure")
        raise
    async = None
    if downloaded and auto_install:
        async = install_updates.delay()
        logger.debug("Auto starting install of %d packages in task %s" %
                     (len(downloaded), async.task_id))
    else:
        logger.debug("Finished downloading %d packages" % len(downloaded))
    return downloaded, async


def _do_the_install():
    """This function is expected to be run from a daemonized process"""
    from iondb.rundb import models
    try:
        import ion_tsconfig.TSconfig
        tsconfig = ion_tsconfig.TSconfig.TSconfig()
        enable_security_update = models.GlobalConfig.get().enable_auto_security
        tsconfig.set_securityinstall(enable_security_update)

        success = tsconfig.TSexec_update()
        if success:
            tsconfig.set_state('F')
            models.Message.success("Upgrade completed successfully!")
        else:
            tsconfig.set_state('IF')
            models.Message.error("Upgrade failed during installation.")
        models.Message.objects.filter(expires="system-update-finished").delete()
        models.Message.objects.filter(tags__contains="new-upgrade").delete()
    except Exception as err:
        models.GlobalConfig.objects.update(ts_update_status="Install failure")
        raise
    finally:
        # This will start celeryd if it is not running for any reason after
        # attempting installation.
        call('service', 'celeryd', 'start')
        call('service', 'celerybeat', 'start')


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
@app.task(queue = "diskutil", soft_time_limit = 60)
def update_diskusage(fs_name):
    from iondb.utils.files import percent_full
    from iondb.rundb import models
    fs = models.FileServer.objects.get(name = fs_name)
    if os.path.exists(fs.filesPrefix):
        try:
            fs.percentfull=percent_full(fs.filesPrefix)
            fs.save()
            #logger.debug("Used space: %s %0.2f%%" % (fs.filesPrefix,fs.percentfull))
        except Exception as e:
            logger.warning ("could not update size of %s" % fs.filesPrefix)
            logger.error(e)
    else:
        logger.warning("directory does not exist on filesystem: %s" % fs.filesPrefix)
        fs.percentfull=0
        fs.save()
    return fs_name


@task(queue = 'diskutil')
def post_update_diskusage(fs_name):
    '''Handler for update_diskusage task output'''
    from iondb.rundb import models
    
    inode_threshold = 0.90
    critical_threshold = 99
    warning_threshold = 95
    friendly_threshold = 70
    fs = models.FileServer.objects.get(name = fs_name)

    #========================================================================
    # TS-6669: Banner Message when disk usage gets critical
    #========================================================================
    crit_tag = "%s_disk_usage_critical" % (fs.name)
    warn_tag = "%s_disk_usage_warning" % (fs.name)
    golink = "<a href='%s' >  Visit Data Management</a>" % ('/data/datamanagement/')
    if fs.percentfull > critical_threshold:
        msg = "* * * CRITICAL! %s: Partition is getting very full - %0.2f%% * * *" % (fs.filesPrefix,fs.percentfull)
        logger.debug(msg+"   %s" % golink)
        message  = models.Message.objects.filter(tags__contains=crit_tag)
        if not message:
            models.Message.error(msg+"   %s" % golink,tags=crit_tag)
            notify_diskfull(msg)
    elif fs.percentfull > warning_threshold:
        msg = "%s: Partition is getting full - %0.2f%%" % (fs.filesPrefix,fs.percentfull)
        logger.debug(msg+"   %s" % golink)
        message  = models.Message.objects.filter(tags__contains=warn_tag)
        if not message:
            models.Message.error(msg+"   %s" % golink,tags=warn_tag)
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
        gc = models.GlobalConfig.get()
        auto_action_enabled = gc.auto_archive_enable
        if not auto_action_enabled and fs.percentfull > friendly_threshold:
            msg = "Data Management Auto Actions are not enabled and %s is %0.2f%% full" % (fs.filesPrefix,fs.percentfull)
            logger.debug(msg+"   %s" % golink)
            message  = models.Message.objects.filter(tags__contains=friendly_tag)
            if not message:
                models.Message.error(msg+"   %s" % golink,tags=friendly_tag, route='_StaffOnly')
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
            message  = models.Message.objects.filter(tags__contains=inode_tag)
            if not message:
                models.Message.error(msg, tags=inode_tag, route='_StaffOnly')
                notify_diskfull(msg)
        else:
            models.Message.objects.filter(tags__contains=inode_tag).delete()
    except:
        logger.error(traceback.format_exc())


# Expires after 5 minutes; is scheduled every 10 minutes
# To trigger celery task from command line:
# python -c 'import iondb.bin.djangoinit, iondb.rundb.tasks as tasks; tasks.check_disk_space.apply_async()'
@periodic_task(run_every=600, expires=300, queue="periodic", ignore_result = True)
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
            async_result = update_diskusage.apply_async([fs.name], link = post_update_diskusage.s())
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
            message  = models.Message.objects.filter(tags__contains=warn_tag)
            if not message:
                models.Message.warn(msg, tags=warn_tag)
        else:
            models.Message.objects.filter(tags__contains=warn_tag).delete()
            #logger.debug("Root partition is spacious enough with %0.2f" % (root_free_space))
    except:
        logger.error(traceback.format_exc())


def notify_diskfull(msg):
    '''sends an email with message'''
    from django.core import mail
    #TODO make a utility function to send email
    try:
        recipient = models.User.objects.get(username='dm_contact').email
        logger.warning("dm_contact is %s." % recipient)
    except:
        logger.warning("Could not retrieve dm_contact.  No email sent.")
        return False

    # Check for blank email
    # TODO: check for valid email address
    if recipient is None or recipient == "":
        logger.warning("No dm_contact email configured.  No email sent.")
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
        recipient = recipient.replace(',',' ').replace(';',' ').split()
        logger.debug(recipient)
        mail.send_mail(subject_line, message, reply_to, recipient)
    except:
        logger.warning(traceback.format_exc())
        return False
    else:
        logger.info("Notification email sent for user acknowledgement")
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


@app.task
def install_reference(args, reference_id):
    from iondb.rundb import models
    full_path, monitor_id = args
    monitor = models.FileMonitor.objects.get(id=monitor_id)
    reference = models.ReferenceGenome.objects.get(id=reference_id)
    reference.status = "preprocessing"
    reference.save()
    extracted_path = os.path.join(monitor.local_dir, "reference_contents")
    extract_zip(full_path, extracted_path)
    reference.reference_path = extracted_path
    if reference.index_version == settings.TMAP_VERSION:
        reference.status = "complete"
        reference.enabled = True
        reference.enable_genome()
    else:
        reference.status = "queued"
        result = build_tmap_index.delay(reference.id)
        reference.celery_task_id = result.task_id
    reference.save()


@app.task
def get_raid_stats():
    raidCMD = ["/usr/bin/ion_raidinfo"]
    q = subprocess.Popen(raidCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    if q.returncode == 0:
        raid_stats = stdout
    else:
        raid_stats = None
    return raid_stats


@app.task (queue = "w1", soft_time_limit = 60)
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


@task (queue = "w1", ignore_result = True)
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
                models.Message.warn(msg+"   %s" % golink,tags="raid_alert")
        else:
            message.delete()


@periodic_task(run_every=timedelta(minutes=20), expires=600, queue="periodic", ignore_result = True)
def check_raid_status():
    '''check RAID state and alert user with message banner of any problems'''
    get_raid_stats_json.apply_async(link = post_check_raid_status.s())


@app.task
def disk_check_status():
    status, stdout, stderr = call("tune2fs", "-l", "/dev/sda1")
    if status != 0:
        logger.error("tune2fs error: " + stderr)
    return stdout


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
def hide_apt_sources():
    filename = "/etc/apt/sources.list"
    hidden = filename+".hide"
    os.rename(filename, hidden)
    return


@app.task
def restore_apt_sources():
    filename = "/etc/apt/sources.list"
    hidden = filename+".hide"
    os.rename(hidden, filename)
    return


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
    # get Torrent Suite version string.
    from ion import version
    ts_version = version

    # get distribution string
    def get_distrib_name():
        with open('/etc/lsb-release','r') as fp:
            for line in fp.readlines():
                if line.startswith('DISTRIB_CODENAME'):
                    return line.split('=')[1].strip()
        return 'lucid'  # failsafe default which may work

    os_codename = get_distrib_name()

    if enable:
        find_string = "updates\/software.*%s\/" % (os_codename)
        replace_string = "updates\/software\/archive %s\/" % (ts_version)
    else:
        find_string = "updates\/software\/archive %s\/" % (ts_version)
        replace_string = "updates\/software %s\/" % (os_codename)
    sed_string = "s/%s/%s/g" % (find_string, replace_string)

    # Possible locations of Ion Apt repository strings:
    #     /etc/apt/sources.list
    #     /etc/apt/sources.list.d/*.list
    #
    filepaths = [os.path.join("/etc/apt/sources.list.d",x) for x in os.listdir("/etc/apt/sources.list.d") if os.path.splitext(x)[1] == '.list']
    filepaths.append("/etc/apt/sources.list")
    for filepath in filepaths:
        logger.debug("Looking in %s" % filepath)
        cmd = ["sed", "-i", sed_string, filepath]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
        except:
            logger.error("%s: cmd = '%s'" % (sys._getframe().f_code.co_name, cmd))
        if proc.returncode:
            logger.error("%s: %s" % (sys._getframe().f_code.co_name, stderr))


@app.task(queue = 'w1')
def post_run_nodetests(result):
    '''Handler for return value of set of tasks running test_node_and_update_db
    result is an array of strings
    '''
    from iondb.rundb.models import Message
    logger.info(result)
    
    if 'error' in  result:
        cluster_status = 'error'
    elif 'warning' in result:
        cluster_status = 'warning'
    else:
        cluster_status = 'good'
        
    if cluster_status:
        message = Message.objects.filter(tags="cluster_alert")
        if 'error' in cluster_status or 'warning' in cluster_status:
            if not message:
                msg = 'WARNING: Cluster node failure.'
                golink = "<a href='%s' >  Visit Services Tab  </a>" % ('/configure/services/')
                Message.warn(msg+"   %s" % golink,tags="cluster_alert")
        else:
            message.delete()


@task(queue = 'w1', soft_time_limit = 60)
def test_node_and_update_db(node, head_versions):
    '''run tests'''
    from iondb.rundb.configure.cluster_info import connect_nodetest, config_nodetest
    
    logger.info("Testing node: %s" % node.name)
    node_status = connect_nodetest(node.name)
    if node_status['status'] == 'good':
        logger.info("Node: %s passed connect test" % node.name)
        node_status.update(config_nodetest(node.name, head_versions))
        
    try:
        add_eventlog(node, node_status)
    except:
        # TODO
        pass
    
    # update cruncher database entry
    node.state = node_status['status'][0].upper()
    node.info = node_status
    node.save()
    
    return node_status['status']


@periodic_task(run_every=timedelta(minutes=20), expires=600, queue="periodic", ignore_result = True)
def check_cluster_status():
    # run tests for cluster nodes and alert user with message banner of any problems
    from ion.utils.TSversion import findVersions
    from iondb.rundb.models import Cruncher
    
    nodes = Cruncher.objects.all().order_by('pk')
    if not nodes:
        pass
    else:
        # launch parallel celery tasks to test all nodes, and then call the final task to return a status
        # Note that parallelism will be limited by the number of workers in a queue.
        head_versions, _ = findVersions()
        tasks = group(test_node_and_update_db.s(node, head_versions) for node in nodes)
        c = chord(tasks)(post_run_nodetests.s())
        return c


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
    
    addlog = addlog or node.info.get('error','') != new_status.get('error','')
    addlog = addlog or node.info.get('version_test_errors', '') != new_status.get('version_test_errors', '')
    msg += "Error: %s %s<br>" % (new_status.get('error', ''), new_status.get('version_test_errors', ''))

    old_test_results = dict(node.info.get('config_tests', []) + node.info.get('connect_tests', []))
    for test_name, test_result in new_status.get('config_tests', []) + new_status.get('connect_tests', []):
        msg += "%s: %s<br>" % (test_name, test_result)
        addlog = addlog or old_test_results.get(test_name, '') != test_result
    
    if addlog:
        EventLog.objects.add_entry(node, msg, username="system")
