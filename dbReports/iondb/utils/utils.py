# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import collections
import datetime
import logging
from django.conf import settings
from distutils.version import StrictVersion
import csv
import re
import os
import math
import subprocess
import apt

logger = logging.getLogger(__name__)


def convert(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


def validate_csv_template_version(headerName=None, isSampleCSV=None, isPlanCSV=None, firstRow=None):
    csv_version_row = firstRow
    COLUMN_CSV_VERSION = headerName

    failed = []
    isToSkipRow = False
    isToAbort = False
    isValidCSV = True
    user_SampleCSV_version = None
    if isSampleCSV:
        systemCSV_version = settings.SAMPLE_CSV_VERSION
        systemSupportedCSV_version = settings.SUPPORTED_SAMPLE_CSV_VERSION
        csvTemplate = "Sample"
    if isPlanCSV:
        systemCSV_version = settings.PLAN_CSV_VERSION
        systemSupportedCSV_version = settings.SUPPORTED_PLAN_CSV_VERSION
        csvTemplate = "Plan"
    logger.debug("iondb.utils.validate_csv_template_version %s" % csv_version_row)
    # skip this row if no values found (will not prohibit the rest of the files from upload
    if not any(list(csv_version_row)):
        isToAbort = True
    else:
        csv_verSampleObj = csv.DictReader(csv_version_row)
        try:
            for row in csv_verSampleObj:
                if COLUMN_CSV_VERSION in row:
                    user_SampleCSV_version = row[COLUMN_CSV_VERSION]
                else:
                    isToAbort = True
        except Exception, e:
            logger.debug("iondb.utils.utils.py Unknown exception %s" % e)

        if user_SampleCSV_version:
            try:
                if re.findall(r"^\d$", str(user_SampleCSV_version)):
                    user_SampleCSV_version = float(user_SampleCSV_version)
                user_SampleCSV_version = str(user_SampleCSV_version)
                systemCSV_version = str(systemCSV_version)
                if user_SampleCSV_version not in systemSupportedCSV_version:
                    isToAbort = True
                    isValidCSV = False
            except Exception as e:
                isToAbort = True
                isValidCSV = False
        else:
            isToAbort = True

    if isToAbort and not isValidCSV:
        failed.append((COLUMN_CSV_VERSION, "%s CSV Version(%s) is not supported. Please download the current CSV %s upload file and try again" % (csvTemplate, user_SampleCSV_version, csvTemplate)))
        return failed, isToSkipRow, isToAbort
    elif isToAbort:
        failed.append((COLUMN_CSV_VERSION, "%s CSV Version is missing. Please download the current CSV %s upload file and try again" % (csvTemplate, csvTemplate)))
        return failed, isToSkipRow, isToAbort

    return failed, isToSkipRow, isToAbort


def bytesToHumanReadableSize(size):
    """
    Helper method which will convert bytes to a human readable string
    :param size: The size in bytes
    :return: A human readable string with the correct units
    """

    size_name = ("KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size, 1024)))
    p = math.pow(1024, i)
    s = round(size/p, 2)
    return '%s %s' % (s, size_name[i-1]) if s > 0 else '0B'


def directorySize(directory):
    """
    Gets the size of a directories contents recursively
    :param directory:
    :return: The total size of all the files in bytes
    """

    totalSize = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            totalSize += os.path.getsize(fp)
    return totalSize


def getPackageName(filepath):
    """
    Will get the aptitiude package name which the file is under.
    :param filepath:
    :return:
    """
    try:
        cmd = subprocess.Popen(['dpkg', '-S', filepath], stdout=subprocess.PIPE)
        result = cmd.communicate()[0]
        return result.split(':')[0]
    except Exception:
        logger.exception('Failed to run dpkg.')
        return ''


class VersionChange():

    """
    This class encompasses the changes for a given version
    """

    # the name of the package
    PackageName = ''

    # the version number
    Version = ''

    # the urgency of the update
    Urgency = ''

    # the changes to this version
    Changes = list()

    # the maintainer of the version
    Maintainer = ''

    # the email address of the contact
    Email = ''

    # the build time of the package
    BuildTime = datetime.datetime.now()

    # the distribution for the package
    Distribution = ''

    @staticmethod
    def ParseChangeLog(changelog):
        """
        Parse the changelog and create a dictionary of changes
        """
        # create a dictionary for returning
        changes = dict();

        # break it up into lines
        lines = changelog.split('\n')

        # generate a list of all of the start lines
        startLines = list()
        for currentLineIndex in range(len(lines)):
            if 'urgency' in lines[currentLineIndex]:
                startLines.append(currentLineIndex)
        startLines.append(len(lines))

        for startLineIndex in range(len(startLines) - 1):
            startIndex = startLines[startLineIndex]
            endIndex = startLines[startLineIndex + 1]
            block = lines[startIndex:endIndex]

            # parse header bits
            bits = block[0].split(' ')
            change = VersionChange()
            change.Package = bits[0].strip()
            change.Version = bits[1].strip('(').strip(')')
            change.Distribution = bits[2].strip().strip(';')
            change.Urgency = bits[3].split('=')[1].strip()

            # parse footer bits
            bits = block[-3]
            change.Maintainer = bits.split('<')[0].strip('-- ')
            change.Email = bits.split('<')[1].split('>')[0]
            change.BuildTime = bits.split('>')[1].strip()
            change.Changes = block[2:-4]

            # append this object to the dictionary
            changes[change.Version] = change

        return changes


def GetChangeLog(packageName, versionName):
    """
    This will return the change log information for a given package and version
    :param packageName: The name of the plugin
    :param versionName: The version in question
    :return: The changes in the changelog
    """

    # check to make sure we have a url which we can attempt to use
    if not hasattr(settings, 'PLUGIN_CHANGELOG_URL'):
        return VersionChange()

    cache = apt.Cache()

    if not cache.has_key(packageName):
        return VersionChange()

    package = cache[packageName]
    if not versionName in package.versions:
        return VersionChange()

    # get the changelog for parsing
    changelog = package.get_changelog(uri=settings.PLUGIN_CHANGELOG_URL)
    changes = VersionChange.ParseChangeLog(changelog)
    return changes[versionName] if changes.has_key(versionName) else VersionChange()


def get_apt_cache(packageName):
    cache = apt.Cache()
    package = cache[packageName]
    return package, cache


def cidr_lookup(address):
    # Lookup table of netmask values and the corresponding CIDR mask bit.
    maskbits = (
        ("128.0.0.0", 1),
        ("192.0.0.0", 2),
        ("224.0.0.0", 3),
        ("240.0.0.0", 4),
        ("248.0.0.0", 5),
        ("252.0.0.0", 6),
        ("254.0.0.0", 7),
        ("255.0.0.0", 8),
        ("255.128.0.0", 9),
        ("255.192.0.0", 10),
        ("255.224.0.0", 11),
        ("255.240.0.0", 12),
        ("255.248.0.0", 13),
        ("255.252.0.0", 14),
        ("255.254.0.0", 15),
        ("255.255.0.0", 16),
        ("255.255.128.0", 17),
        ("255.255.192.0", 18),
        ("255.255.224.0", 19),
        ("255.255.240.0", 20),
        ("255.255.248.0", 21),
        ("255.255.252.0", 22),
        ("255.255.254.0", 23),
        ("255.255.255.0", 24),
        ("255.255.255.128", 25),
        ("255.255.255.192", 26),
        ("255.255.255.224", 27),
        ("255.255.255.240", 28),
        ("255.255.255.248", 29),
        ("255.255.255.252", 30),
        ("255.255.255.254", 31),
        ("255.255.255.255", 32),
        )
    for (netmask, maskbit) in maskbits:
        if address == netmask:
            return maskbit


def is_TsVm():
    # returns True if the TS is running as a VM instance on S5
    return os.path.exists('/etc/init.d/mountExternal')


def is_internal_server():
    isInternalServer = False  
    try:
        isInternalServer = os.path.exists("/opt/ion/.ion-internal-server")
    except:
        logger.exception("Failed to create isInternalServer variable")
    return isInternalServer

