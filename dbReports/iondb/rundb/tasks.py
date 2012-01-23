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
import urllib2
import os
import string
import random
import subprocess
from math import floor
from django.conf import settings
import zipfile
import os.path
import sys

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

def unzip_archive(root, data):
    if not os.path.exists(root):
        os.mkdir( root, 0777)
    zip_file = zipfile.ZipFile(data, 'r')
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
    directories = [os.path.dirname(f) for f in files]
    for directory in directories:
        path = os.path.join(root, directory)
        if not os.path.exists(path):
            os.makedirs(path)

@task
def test_task(args):
    logger = test_task.get_logger()
    import getpass
    myUser = getpass.getuser()
    logger.info("Test Task Running")
    return "%s, brought to you by Celery" % myUser


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
            while True:
                chunk = req.read(CHUNK)
                downloaded += len(chunk)
                #print floor( (downloaded / total_size) * 100 )
                if not chunk: break
                fp.write(chunk)
    except urllib2.HTTPError, e:
        print "HTTP Error:",e.code , url
        return False
    except urllib2.URLError, e:
        print "URL Error:",e.reason , url
        return False
    except:
        print "Other Error"
        return False

    return file


@task
def downloadGenome(url, genomeID):
    """download a genome, and update the genome model"""
    downloadChunks(url)


@task
def downloadPlugin(url, plugin):
    """download a plugin and unzip that it"""
    downloaded = downloadChunks(url)
    plugin.status["installStatus"] = "downloading"
    plugin.save()

    if not downloaded:
        plugin.status["installStatus"] = "failed"
        plugin.status["result"] = "processed"
        plugin.save()
        return False

    plugin.status["installStatus"] = "installed"

    #for now rename it to the name of the zip file
    plugin.name = os.path.splitext(os.path.basename(url))[0]
    plugin.path = os.path.join(settings.PLUGIN_PATH, plugin.name )
    plugin.save()
    unzipStatus = unzip_archive(plugin.path, downloaded)
    #clean up
    os.unlink(downloaded)
    os.rmdir(os.path.dirname(downloaded))
    if unzipStatus:
        plugin.status["result"] = "unzipped"
    else:
        plugin.status["result"] = "failed to unzip"
    plugin.save()


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
        otStatus = open("/tmp/OTstatus",'w')
        updateStatus = findHosts.findOneTouches()
        otStatus.write(str(updateStatus))
        otStatus.close()
        #now remove the lock
        os.unlink("/tmp/OTlock")
        os.unlink("/tmp/OTstatus")
        return True

    return False


