# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Tasks
=====

The ``tasks`` module contains all the Python functions which spawn Celery
tasks in the background.

Not all functions contained in ``tasks`` are actual Celery tasks, only those
that have the  ``@task`` decorator.
"""

from __future__ import division

from celery.task import task
from celery.task import periodic_task
import urllib2
import os
import string
import random
import subprocess
import shutil
from django.conf import settings
import zipfile
import os.path
import sys
import re
import json
import models
import logging
from datetime import timedelta
from iondb.plugins.manager import PluginManager

import urlparse

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


def unzip(dir, file):
    extractDir = os.path.join(settings.PLUGIN_PATH, os.path.splitext(file)[0])
    if not os.path.exists(extractDir):
        os.mkdir( extractDir, 0777)
    zfobj = zipfile.ZipFile(os.path.join(dir,file))
    for name in zfobj.namelist():
        if name.endswith('/'):
            #make dirs
            if not os.path.exists(os.path.join(extractDir, name)):
                os.mkdir(os.path.join(extractDir, name))
        else:
            outfile = open(os.path.join(extractDir, name), 'wb')
            outfile.write(zfobj.read(name))
            outfile.close()
    return True

def unzipPlugin(file):
    """Unzip a file and return the common prefix name, which should be the plugin name"""
    zfobj = zipfile.ZipFile(file)
    namelist = zfobj.namelist()
    prefix, files = get_common_prefix(namelist)
    file_name = file.split(".")[0]

    scratch_path = os.path.join(settings.PLUGIN_PATH,"scratch",file_name)

    if not os.path.exists(scratch_path):
        os.mkdir(scratch_path, 0755)

    for name in zfobj.namelist():
        if name.endswith('/'):
            #make dirs
            if not os.path.exists(os.path.join(scratch_path,name)):
                os.mkdir(os.path.join(scratch_path,name))
        else:
            outfile = open(os.path.join( scratch_path, name), 'wb')
            outfile.write(zfobj.read(name))
            outfile.close()

    #move to the plugin dir
    shutil.move(os.path.join(scratch_path,file_name,prefix), os.path.join(settings.PLUGIN_PATH ) )

    # Plugin folders are owned by root and not writable by plugin user.
    # zip extraction may not preserve executable permissions
    return prefix


def unzip_archive(root, data):
    if not os.path.exists(root):
        os.mkdir( root, 0777)
    zip_file = zipfile.ZipFile(data, 'r', allowZip64=True)
    namelist = zip_file.namelist()
    namelist = valid_files(namelist)
    prefix, files = get_common_prefix(namelist)
    make_relative_directories(root, files)
    out_names = [(n, f) for n, f in zip(namelist, files) if
                        os.path.basename(f) != '']
    for key, out_name in out_names:
        if os.path.basename(out_name) != "":
            full_path = os.path.join(root, out_name)
            contents = zip_file.open(key, 'r')
            try:
                output_file = open(full_path, 'wb')
                output_file.write(contents.read())
                output_file.close()
            except IOError as err:
                print("For zip's '%s', could not open '%s'" % (key, full_path))
    return [f for n, f in out_names]


def get_common_prefix(files):
    """For a list of files, a common path prefix and a list file names with
    the prefix removed.

    Returns a tuple (prefix, relative_files):
        prefix: Longest common path to all files in the input. If input is a
                single file, contains full file directory.  Empty string is
                returned of there's no common prefix.
        relative_files: String containing the relative paths of files, skipping
                        the common prefix.
    """
    # Handle empty input
    if not files or not any(files):
        return '', []
    # find the common prefix in the directory names.
    directories = [os.path.dirname(f) for f in files]
    prefix = os.path.commonprefix(directories)
    start = len(prefix)
    if all(f[start] == "/" for f in files):
        start += 1
    relative_files = [f[start:] for f in files]
    return prefix, relative_files


def valid_files(files):
    black_list = [lambda f: "__MACOSX" in f]
    absolute_paths = [os.path.isabs(d) for d in files]
    if any(absolute_paths) and not all(absolute_paths):
        raise ValueError("Archive contains a mix of absolute and relative paths.")
    return [f for f in files if not any(reject(f) for reject in black_list)]


def make_relative_directories(root, files):
    directories = ( os.path.dirname(f) for f in files )
    for directory in directories:
        path = os.path.join(root, directory)
        if not os.path.exists(path):
            os.makedirs(path)

@task
def delete_that_folder(directory, message):
    logger = delete_that_folder.get_logger()
    def delete_error(func, path, info):
        logger.error("Failed to delete %s: %s" %  (path, message))
    logger.info("Deleting %s" % directory)
    shutil.rmtree(directory, onerror=delete_error)

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
    except urllib2.HTTPError, e:
        print "HTTP Error:",e.code , url
        delete_that_folder(temp_path_uniq, "after download error")
        return False
    except urllib2.URLError, e:
        print "URL Error:",e.reason , url
        delete_that_folder(temp_path_uniq, "after download error")
        return False
    except:
        print "Other Error"
        delete_that_folder(temp_path_uniq, "after download error")
        return False

    return file

@task
def downloadGenome(url, genomeID):
    """download a genome, and update the genome model"""
    downloadChunks(url)


import zeroinstallHelper

# Helper for downloadPlugin task
def downloadPluginZeroInstall(url, plugin):
    """ To be called for zeroinstall xml feed urls.
        Returns plugin prototype, not full plugin model object.
    """
    try:
        downloaded  = zeroinstallHelper.downloadZeroFeed(url)
        feedName = zeroinstallHelper.getFeedName(url)
    except:
        plugin.status["installStatus"] = "failed"
        plugin.status["result"] = str(sys.exc_info()[1][0])
        return False

    # The url field stores the zeroinstall feed url
    plugin.url = url

    if not downloaded:
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
    else:
        # Plugin expanded without top level folder
        plugin.path = downloaded
        # assert launch.sh exists?

    plugin.status["result"] = "0install"
    plugin.name = feedName.replace(" ","")
    # Other fields we can get from zeroinstall feed?

    # Version is parsed during install - from launch.sh, ignoring feed value
    return plugin

# Helper for downloadPlugin task
def downloadPluginArchive(url, plugin):
    downloaded = downloadChunks(url)
    if not downloaded:
        plugin.status["installStatus"] = "failed"
        plugin.status["result"] = "failed to download '%s'" % url
        return False

    # ZIP file must named with plugin name - fragile
    # FIXME - handle (1) additions (common for multiple downloads via browser)
    # FIXME - handle version string in ZIP archive name
    #     Suggested fix - extract to temporary directory, find plugin name,
    #     rename to final location
    plugin.name = os.path.splitext(os.path.basename(url))[0]
    plugin.path = os.path.join(settings.PLUGIN_PATH, plugin.name )

    unzipStatus = unzip_archive(plugin.path, downloaded)

    #clean up archive file and temp dir (archive should be only file in dir)
    os.unlink(downloaded)
    os.rmdir(os.path.dirname(downloaded))

    if unzipStatus:
        plugin.status["result"] = "unzipped"
    else:
        plugin.status["result"] = "failed to unzip"

    plugin.status["installStatus"] = "installed"

    return True

@task
def downloadPlugin(url, plugin, zipFile=None):
    """download a plugin, extract and install it"""

    logger = downloadPlugin.get_logger()

    #normalise the URL
    url = urlparse.urlsplit(url).geturl()

    if not zipFile:
        plugin.status["installStatus"] = "downloading"

        if url.endswith(".xml"):
            status = downloadPluginZeroInstall(url, plugin)
            logger.error("xml") # logfile
        else:
            status = downloadPluginArchive(url, plugin)
            logger.error("zip") # logfile

        if not status:
            # FIXME - Errors!
            installStatus = plugin.status.get('installStatus', 'Unknown')
            result = plugin.status.get('result', 'unknown')
            msg = "Plugin install '%s', Result: '%s'" % (installStatus, result)

            logger.error(msg) # logfile
            models.Message.error(msg) # UI message
            return False
    else:
        # Extract zipfile
        scratch_path = os.path.join(settings.PLUGIN_PATH,"scratch")
        zip_file = os.path.join(scratch_path, zipFile)
        plugin.status["installStatus"] = "extracting zip"

        plugin.name = unzipPlugin(zip_file)

        plugin.path = os.path.join(settings.PLUGIN_PATH, plugin.name)
        plugin.status["installStatus"] = "installing from zip"

        #remove the zip file
        os.unlink(zipFile)

    # Now that it has been downloaded,
    # convert pre-plugin into real db plugin object
    pm = PluginManager()
    (new_plugin, updated) = pm.install(plugin.name, plugin.path)

    # Copy over download status messages and url
    new_plugin.status = plugin.status
    if plugin.url:
        new_plugin.url = plugin.url
    new_plugin.save()

    logger.info("Successfully downloaded and installed plugin %s v%s from '%s'", new_plugin.name, new_plugin.version, url)

    return new_plugin

@task
def contact_info_flyaway():
    """This invokes an external on the path which performs 3 important steps:
        Pull contact information from the DB
        Black magic
        Axeda has the contact information
    """
    logger = contact_info_flyaway.get_logger()
    logger.info("The user updated their contact information.")
    cmd = ["/opt/ion/RSM/updateContactInfo.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("updateContactInfo.py output error information:\n%s" % stderr)
    return stdout


@task
def static_ip(address, subnet, gateway):
    """Usage: TSstaticip [options]
         --ip      Define host IP address
         --nm      Define subnet mask (netmask)
         --nw      Define network ID
         --bc      Define broadcast IP address
         --gw      Define gateway/router IP address
    """
    logger = static_ip.get_logger()
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


@task
def dhcp():
    """Usage: TSstaticip [options]
        --remove  Sets up dhcp, removing any static IP settings
    """
    logger = dhcp.get_logger()
    cmd = ["/usr/sbin/TSstaticip", "--remove"]
    logger.info("Network: Setting host DHCP, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@task
def proxyconf(address, port, username, password):
    """Usage: TSsetproxy [options]
         --address     Proxy address (http://proxy.net)
         --port         Proxy port number
         --username    Username for authentication
         --password    Password for authentication
         --remove      Removes proxy setting
    """
    logger = proxyconf.get_logger()
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


@task
def ax_proxy():
    """Usage: TSsetproxy [options]
         --remove      Removes proxy setting
    """
    logger = ax_proxy.get_logger()
    cmd = ["/usr/sbin/TSsetproxy", "--remove"]
    logger.info("Network: Removing proxy settings, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@task
def dnsconf(dns):
    """Usage: TSsetproxy [options]
         --remove      Removes proxy setting
    """
    logger = ax_proxy.get_logger()
    cmd = ["/usr/sbin/TSdns", "--dns", dns]
    logger.info("Network: Changing DNS settings, '%s'" % " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stderr:
        logger.warning("Network error: %s" % stderr)
    return stdout


@task
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


@task
def build_tmap_index(reference, read_sample_size=None):
    """ Provides a way to kick off the tmap index generation
        this should spawn a process that calls the build_genome_index.pl script
        it may take up to 3 hours.
        The django server should contacts this method from a view function
        When the index creation processes has exited, cleanly or other wise
        a callback will post to a url that will update the record for the library data
        letting the genome manager know that this now exists
        until then this genome will be listed in a unfinished state.
    """

    logger = build_tmap_index.get_logger()
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
    if read_sample_size is not None:
        cmd.append("--read-sample-size")
        cmd.append(read_sample_size)

    ret, stdout, stderr = call(*cmd, cwd=settings.TMAP_DIR)
    if ret == 0:
        logger.debug("Successfully built the TMAP %s index for %s" %
                    (settings.TMAP_VERSION, reference.short_name))
        reference.status = 'created'
        reference.enabled = True
        reference.index_version = settings.TMAP_VERSION
        reference.reference_path = os.path.join(settings.TMAP_DIR, reference.short_name)
    else:
        logger.error('TMAP index rebuild "%s" failed:\n%s' %
                     (" ".join(cmd), stderr))
        reference.status = 'error'
        reference.verbose_error = json.dumps((stdout, stderr, ret))
    reference.save()

    return ret == 0


def IonReporterWorkflows(autorun=True):

    try:
        if autorun:
            IonReporterUploader= models.Plugin.objects.get(name="IonReporterUploader",selected=True,active=True,autorun=True)
        else:
            IonReporterUploader= models.Plugin.objects.get(name="IonReporterUploader",selected=True,active=True)

        logging.error(IonReporterUploader.config)
        config = IonReporterUploader.config
    except models.Plugin.DoesNotExist:
        error = "IonReporterUploader Plugin Not Found."
        logging.error(error)
        return False, error

    try:
        headers = {"Authorization" : config["token"] }
        url = config["protocol"] + "://" + config["server"] + ":" + config["port"] +"/grws/analysis/wflist"
        logging.info(url)
    except KeyError:
        error = "IonReporterUploader Plugin Config is missing needed data."
        logging.error(error)
        return False, error

    try:
        #using urllib2 right now because it does NOT verify SSL certs
        req = urllib2.Request(url = url, headers = headers)
        response = urllib2.urlopen(req)
        content = response.read()
        content = json.loads(content)
        workflows = content["workflows"]
        return True, workflows
    except:
        error = "IonReporterUploader could not contact the server."
        logging.error(error)
        return False, error


@periodic_task(run_every=timedelta(days=1))
def scheduled_update_check():
    logger = scheduled_update_check.get_logger()
    try:
        check_updates.delay()
    except Exception as err:
        logger.error("TSconfig raised '%s' during a scheduled update check." % err)
        gconfig = models.GlobalConfig.get()
        gconfig.ts_update_status = "Update failure"
        gconfig.save()
        raise


@task
def check_updates():
    """Currently this is passed a TSConfig object; however, there might be a
    smoother design for this control flow.
    """
    logger = check_updates.get_logger()
    try:
        import ion_tsconfig.TSconfig
        tsconfig = ion_tsconfig.TSconfig.TSconfig()
        packages = tsconfig.TSpoll_pkgs()
    except Exception as err:
        logger.error("TSConfig raised '%s' during update check." % err)
        gconfig = models.GlobalConfig.get()
        gconfig.ts_update_status = "Update failure"
        gconfig.save()
        raise
    async = None
    if packages and tsconfig.get_autodownloadflag():
        async = download_updates.delay()
        logger.debug("Auto starting download of %d packages in task %s" %
                     (len(packages), async.task_id))
    return packages, async


@task
def download_updates(auto_install=False):
    logger = download_updates.get_logger()
    try:
        import ion_tsconfig.TSconfig
        tsconfig = ion_tsconfig.TSconfig.TSconfig()
        downloaded = tsconfig.TSexec_download()
    except Exception as err:
        logger.error("TSConfig raised '%s' during a download" % err)
        gconfig = models.GlobalConfig.get()
        gconfig.ts_update_status = "Download failure"
        gconfig.save()
        raise
    async = None
    if downloaded and auto_install:
        async = install_updates.delay()
        logger.debug("Auto starting install of %d packages in task %s" %
                     (len(downloaded), async.task_id))
    return downloaded, async


def _do_the_install():
    """This function is expected to be run from a daemonized process"""
    try:
        import ion_tsconfig.TSconfig
        tsconfig = ion_tsconfig.TSconfig.TSconfig()
        success = tsconfig.TSexec_update()
        if success:
            tsconfig.set_state('F')
            models.Message.success("Upgrade completed successfully!")
        else:
            tsconfig.set_state('IF')
            models.Message.error("Upgrade failed during installation.")
    except Exception as err:
        gconfig = models.GlobalConfig.get()
        gconfig.ts_update_status = "Install failure"
        gconfig.save()
        raise


@task
def install_updates():
    try:
        run_as_daemon(_do_the_install)
    except Exception as err:
        gconfig = models.GlobalConfig.get()
        gconfig.ts_update_status = "Install failure"
        gconfig.save()
        raise
