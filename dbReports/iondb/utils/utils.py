# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import collections
import csv
import datetime
import logging
import math
import os
import pwd
import re
import subprocess
import traceback
import types
from contextlib import contextmanager
from multiprocessing import Pool
import requests
import json
import urlparse
import urllib2
import base64

import apt
from dateutil.parser import parse as parse_date
from django.conf import settings
from django.utils.translation import ugettext_lazy
from django.http import HttpResponse

logger = logging.getLogger(__name__)

TIMEOUT_LIMIT_SEC = settings.REQUESTS_TIMEOUT_LIMIT_SEC

def convert(data):
    if isinstance(data, str):
        return str(data)
    elif isinstance(data, unicode):
        return unicode(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, iter(data.items())))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


def validate_csv_template_version(
    headerName=None,
    isSampleCSV=None,
    isPlanCSV=None,
    firstRow=None,
    SampleCSVTemplateLabel="Sample",
    PlanCSVTemplateLabel="Plan",
):
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
        csvTemplate = SampleCSVTemplateLabel
    if isPlanCSV:
        systemCSV_version = settings.PLAN_CSV_VERSION
        systemSupportedCSV_version = settings.SUPPORTED_PLAN_CSV_VERSION
        csvTemplate = PlanCSVTemplateLabel
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
        except Exception as e:
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
        error_message = ugettext_lazy(
            "validate_csv_template_version.invalidversion"
        ) % {
            "template": csvTemplate,
            "version_column": COLUMN_CSV_VERSION,
            "version": user_SampleCSV_version,
        }  # "%(template)s %(version_column)s (%(version)s) is not supported. Please download the current CSV %(template)s and try again"
        failed.append((COLUMN_CSV_VERSION, error_message))
        return failed, isToSkipRow, isToAbort
    elif isToAbort:
        error_message = ugettext_lazy("validate_csv_template_version.missing") % {
            "template": csvTemplate,
            "version_column": COLUMN_CSV_VERSION,
        }  # %(template)s %(version_column)s is missing. Please download the current CSV %(template)s and try again.
        failed.append((COLUMN_CSV_VERSION, error_message))
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
    s = round(size / p, 2)
    return "%s %s" % (s, size_name[i - 1]) if s > 0 else "0B"


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
        cmd = subprocess.Popen(["dpkg", "-S", filepath], stdout=subprocess.PIPE)
        result = cmd.communicate()[0]
        return result.split(":")[0]
    except Exception:
        logger.exception("Failed to run dpkg.")
        return ""


class VersionChange:

    """
    This class encompasses the changes for a given version
    """

    # the name of the package
    PackageName = ""

    # the version number
    Version = ""

    # the urgency of the update
    Urgency = ""

    # the changes to this version
    Changes = list()

    # the maintainer of the version
    Maintainer = ""

    # the email address of the contact
    Email = ""

    # the build time of the package
    BuildTime = datetime.datetime.now()

    # the distribution for the package
    Distribution = ""

    @staticmethod
    def ParseChangeLog(changelog):
        """
        Parse the changelog and create a dictionary of changes
        """
        # create a dictionary for returning
        changes = dict()

        # break it up into lines
        lines = changelog.split("\n")

        # generate a list of all of the start lines
        startLines = list()
        for currentLineIndex in range(len(lines)):
            if "urgency" in lines[currentLineIndex]:
                startLines.append(currentLineIndex)
        startLines.append(len(lines))

        for startLineIndex in range(len(startLines) - 1):
            startIndex = startLines[startLineIndex]
            endIndex = startLines[startLineIndex + 1]
            block = lines[startIndex:endIndex]

            # parse header bits
            bits = block[0].split(" ")
            change = VersionChange()
            change.Package = bits[0].strip()
            change.Version = bits[1].strip("(").strip(")")
            change.Distribution = bits[2].strip().strip(";")
            change.Urgency = bits[3].split("=")[1].strip()

            # parse footer bits
            bits = block[-3]
            change.Maintainer = bits.split("<")[0].strip("-- ")
            change.Email = bits.split("<")[1].split(">")[0]
            change.BuildTime = bits.split(">")[1].strip()
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
    if not hasattr(settings, "PLUGIN_CHANGELOG_URL"):
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
    return os.path.exists("/etc/init.d/mountExternal")


def is_s5orig():
    # returns True if the TS is running on the initially released S5 sequencer
    if is_TsVm():
        try:
            memTotalGb = (
                os.sysconf("SC_PAGE_SIZE")
                * os.sysconf("SC_PHYS_PAGES")
                / (1024 * 1024 * 1024)
            )
            return memTotalGb < 70
        except Exception:
            logger.error("Unable to determine system memory")
    return False


def is_internal_server():
    isInternalServer = False
    try:
        isInternalServer = os.path.exists("/opt/ion/.ion-internal-server")
    except Exception:
        logger.exception("Failed to create isInternalServer variable")
    return isInternalServer


def send_email(recipient, subject_line, text, html=None):
    """sends an email to recipients"""
    import socket
    from django.core import mail
    from iondb.rundb.models import GlobalConfig

    if not recipient:
        logger.warning("No email recipient for %s" % subject_line)
        return False
    else:
        recipient = recipient.replace(",", " ").replace(";", " ").split()

    # Needed to send email
    settings.EMAIL_HOST = "localhost"
    settings.EMAIL_PORT = 25
    settings.EMAIL_USE_TLS = False

    site_name = GlobalConfig.get().site_name or "Torrent Server"
    hname = socket.getfqdn()

    message = "From: %s (%s)\n\n" % (site_name, hname)
    message += text
    message += "\n"

    if html:
        html_message = "From: %s (<a href=%s>%s</a>)<br>" % (site_name, hname, hname)
        html_message += html
        html_message += "<br>"
    else:
        html_message = ""

    reply_to = "donotreply@iontorrent.com"

    # Send the email
    try:
        if html_message:
            sendthis = mail.EmailMultiAlternatives(
                subject_line, message, reply_to, recipient
            )
            sendthis.attach_alternative(html_message, "text/html")
            sendthis.send()
        else:
            mail.send_mail(subject_line, message, reply_to, recipient)
    except:
        logger.error(traceback.format_exc())
        return False
    else:
        logger.info("%s email sent to %s" % (subject_line, recipient))
        return True


def convert_seconds_to_hhmmss_string(seconds):
    if not seconds:
        return ""
    secs = datetime.timedelta(seconds=seconds)
    return str(secs)


def convert_seconds_to_datetime_string(startDateTime, seconds):
    if not startDateTime or not seconds:
        return ""

    secs = datetime.timedelta(seconds=seconds)
    newDateTime = startDateTime + secs
    return newDateTime


def is_endTime_after_startTime(startDateTime, endDateTime):
    if startDateTime and endDateTime:
        startTime = startDateTime
        endTime = endDateTime

        if type(startDateTime) is types.UnicodeType:
            startTime = parse_date(startDateTime)
        if type(endDateTime) is types.UnicodeType:
            endTime = parse_date(endDateTime)

        startTime = startTime.replace(tzinfo=None)
        endTime = endTime.replace(tzinfo=None)
        return endTime >= startTime
    return True


SERVICE_FPATH = "/etc/torrentserver/ion-services"
TELEMETRY_FPATH = "/etc/torrentserver/telemetry-services"


def service_status(services):
    # returns status of input services, True=up / False=down
    def check_service(name):
        proc = subprocess.Popen(
            ["service", name, "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        return (proc.returncode == 0) and ("running" in stdout or "online" in stdout)

    def complicated_status(filename):
        # for some processes need to figure out whether it's running by using pid from file
        try:
            with open(filename, "r") as f:
                pid = int(f.read())
                subprocess.check_call(
                    "ps %d" % pid,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                return True
        except Exception as err:
            return False

    status = {}

    # some services need 2 ways of checking status to handle Ubuntu 14.04 upstart and Ubuntu 18.04 systemd
    for name in services:
        if name.startswith("celery_"):
            status[name] = complicated_status("/var/run/celery/%s.pid" % name)
        elif name == "gridengine-master":
            status["gridengine-master"] = complicated_status(
                "/var/run/gridengine/qmaster.pid"
            ) or check_service("gridengine-master")
        elif name == "gridengine-exec":
            status["gridengine-exec"] = complicated_status(
                "/var/run/gridengine/execd.pid"
            )
        elif name == "dhcp":
            status["dhcp"] = check_service("isc-dhcp-server")
        elif name == "rabbitmq-server":
            status["rabbitmq-server"] = complicated_status(
                "/var/run/rabbitmq/pid"
            ) or check_service("rabbitmq-server")
        elif name in ["tomcat7", "tomcat8"]:
            status["tomcat"] = complicated_status("/var/run/%s.pid" % name)
        elif name == "postgresql":
            status["postgresql"] = check_service("postgresql") or check_service(
                "postgresql@*"
            )
        else:
            status[name] = check_service(name)

    return status


def services_views(excluded_services=None):
    # instead of celeryd, only list the worker queue
    if not isinstance(excluded_services, list):
        excluded_services = ["postgresql", "apache2", "celeryd"]

    return [p for p in ion_processes() if p not in excluded_services]


def ion_processes():
    processes = ion_services()

    # celery worker queue
    processes.extend(
        [
            "celery_w1",
            "celery_plugins",
            "celery_periodic",
            "celery_slowlane",
            "celery_transfer",
            "celery_diskutil",
        ]
    )

    return processes


def ion_services():
    def is_tomcat_installed(server_name):
        service_script = os.path.join("/etc", "init.d", server_name)
        if os.path.exists(service_script):
            return True
        return False

    def get_default_processes():
        # list in start order as well
        default_processes = [
            "postgresql",
            "apache2",
            "gridengine-master",
            "gridengine-exec",
            "rabbitmq-server",
            "celerybeat",
            "celeryd",
            "ionJobServer",
            "ionCrawler",
            "ionPlugin",
            "DjangoFTP",
            "RSM_Launch",
            "deeplaser",
            "dhcp",
            "ntp",
        ]

        # check tomcat 7 or 8
        tomcat_versions = ["tomcat7", "tomcat8"]
        for tomcat in tomcat_versions:
            if is_tomcat_installed(tomcat):
                default_processes.append(tomcat)
        return default_processes

    processes = read_ion_services(SERVICE_FPATH)

    if not processes:
        processes = get_default_processes()

    return processes


def read_ion_services(fpath):
    if not os.path.exists(fpath):
        return []

    processes = []
    with open(fpath, "r") as fp:
        for line in fp.readlines():
            if line.strip() and not line.startswith("#"):
                processes.append(line.strip())

    return processes


def update_telemetry_services(is_on=True, fpath=TELEMETRY_FPATH):
    # Updates the settings file on the filesystem that is used by
    # ansbile to determine if it should enable the telem services
    if not os.path.exists(fpath):
        logger.error("%s not updated because it does not exist." % fpath)
        return

    with open(fpath + ".temp", "w") as temp_fp:
        with open(fpath, "r") as orig_fp:
            for line in orig_fp.readlines():
                if line.startswith("enable_telemetry"):
                    temp_fp.write("enable_telemetry: %s\n" % str(is_on).lower())
                else:
                    temp_fp.write(line)

    www_data_uid = pwd.getpwnam("www-data").pw_uid
    os.chown(temp_fp.name, www_data_uid, www_data_uid)
    os.rename(temp_fp.name, TELEMETRY_FPATH)


@contextmanager
def ManagedPool(*args, **kwargs):
    """ We need a context manager for multiprocessing.Pool that will always call Pool.close()
        python3 has one, but we need one for python 2 to prevent leaving Pools open and running out of pids.
    """
    pool = Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.close()

def get_instrument_info(rig):
    instr = {
        "name": rig.name,
        "type": rig.type or "PGM"
    }
    if instr["type"] == "Raptor":
        instr["type"] = "S5"

    return instr

def update_platform():
    print("Updating major platform...")
    from iondb.rundb.models import Rig, GlobalConfig
    instruments = []
    rigs = Rig.objects.exclude(host_address="")

    if len(rigs) > 0:
        instruments = [get_instrument_info(rig) for rig in rigs]
        instruments = [inst['type'] for inst in instruments]

    if "S5" not in instruments:
        GlobalConfig.objects.update(majorPlatform="pgm_or_proton_only")
    elif "PGM" in instruments or "Proton" in instruments:
        GlobalConfig.objects.update(majorPlatform="mixed")# S5/PGM/Proton
    elif "S5" in instruments:
        GlobalConfig.objects.update(majorPlatform="s5_only")

def get_deprecation_messages():
    try:
        depreOffcyleLocal = os.path.join(settings.OFFCYCLE_UPDATE_PATH_LOCAL, "deprecation_data.json")
        if os.path.exists(depreOffcyleLocal):
            with open(depreOffcyleLocal, 'r') as fh:
                return json.load(fh)
        else:
            return get_deprecation_json_from_url()
    except:
        return None

def get_deprecation_json_from_url():
    try:
        resp = requests.get(settings.OFFCYLE_DEPRECATION_MSG, timeout=TIMEOUT_LIMIT_SEC)
        resp.raise_for_status()
        return resp.json()
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as err:
        logger.error(
            "get_deprecation_messages timeout or connection errors for {u}: {e}".format(
                e=str(err), u=settings.OFFCYLE_DEPRECATION_MSG
            )
        )
        return None
    except ValueError as decode_err:
        logger.error("get_deprecation_messages JSON decode error: {}".format(str(decode_err)))
        return None

def authenticate_using_urllib2(**kwargs):
    """This method uses urllib2 to handle SSL cert failure, this is mainly for 14.04"""
    try:
        request = urllib2.Request(kwargs.get('base_url'))
        base64string = base64.b64encode('%s:%s' % (kwargs.get('username'), kwargs.get('password')))
        request.add_header("Authorization", "Basic %s" % base64string)
        response = urllib2.urlopen(request, timeout=TIMEOUT_LIMIT_SEC)
        return json.load(response)
    except urllib2.HTTPError as exc:
        raise Exception(exc.code)
    except Exception as Error:
        logger.exception(Error)
        raise Exception("Unknown Error")

def exerted_url_authentication(func):
    """ Decorator to authenticate any url link, handles SSL cert issues"""
    def wrapper(**kwargs):
        try:
            return func(**kwargs)
        except requests.exceptions.SSLError:
            return authenticate_using_urllib2(**kwargs)
        except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
        ) as serverError:
            logger.exception(serverError)
            raise Exception(serverError.response.status_code)
        except Exception as exc:
            logger.exception(exc)
            raise Exception("Unknown Error")
    return wrapper

@exerted_url_authentication
def authenticate_fetch_url(**kwargs):
    response = requests.get(kwargs.get('base_url'), auth=(kwargs.get('username'), kwargs.get('password')), timeout=TIMEOUT_LIMIT_SEC)
    response.raise_for_status()
    return response.json()
