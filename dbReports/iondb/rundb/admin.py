# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

from iondb.rundb.models import *
from iondb.rundb import tasks
from iondb.rundb import tsvm
from iondb.rundb.forms import NetworkConfigForm
from iondb.utils import files
from iondb.utils.utils import is_TsVm
from django.contrib import admin
from django.forms import TextInput, Textarea
from django.forms.models import model_to_dict

from django.template import RequestContext
from django.template.defaultfilters import filesizeformat
from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.sessions.models import Session

from django import http
import httplib2
from tastypie.bundle import Bundle

from subprocess import Popen, PIPE

import os
import socket
import fcntl
import struct
import json
import re
import logging
import urllib
import markdown

from ion.utils.TSversion import findUpdates, findOSversion

logger = logging.getLogger(__name__)


def script(script_text, shell_bool=True):
    """run system commands"""
    p = Popen(args=script_text, shell=shell_bool, stdout=PIPE, stdin=PIPE)
    output, errors = p.communicate()
    return output, errors


def outbound_net_port():
    '''Returns the ethernet device associated with the default route'''
    stdout, stderr = script("/sbin/route|awk '/default/{print $8}'")
    port = stdout.strip()
    return port


def mac_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', outbound_net_port()))
    return ''.join(['%02x:' % ord(char) for char in info[18:24]])[:-1]


def parse_http_proxy_string(proxy_string):
    have_proxy = False
    username = ''
    password = ''
    hostname = ''
    portnum = 0

    try:
        proxy_string = proxy_string.partition('#')[0]	# Remove comments (whole-line or suffixed)
        proxy_string = proxy_string.strip()

        # try this format: http_proxy=http://user:pass@11.22.33.44:55
        exp = re.compile('http://(?P<user>.+):(?P<pass>.+)@(?P<host>.+):(?P<port>.+)')
        match = exp.match(proxy_string)
        if match:
            username = match.group('user')
            password = match.group('pass')
            hostname = match.group('host')
            portnum = int(match.group('port'))
            have_proxy = True
        else:
            # try this format: http_proxy=http://11.22.33.44:55
            exp = re.compile('http://(?P<host>.+):(?P<port>.+)')
            match = exp.match(proxy_string)
            if match:
                hostname = match.group('host')
                portnum = int(match.group('port'))
                have_proxy = True
    except:
        pass

    return have_proxy, username, password, hostname, portnum


def get_http_proxy_from_etc_environment():
    try:
        with open('/etc/environment', 'r') as f:
            for line in f:
                if ('http_proxy' in line):
                    return True, line.partition('=')[2].strip()
        return False, ''
    except:
        return False, ''


def ssh_status(dest_host, dest_port):
    try:
        dest = 'TSConnectivityCheck@' + dest_host

        # Read http_proxy from /etc/environment, not from environment variables.
        have_proxy, proxy_string = get_http_proxy_from_etc_environment()

        if (have_proxy):

            have_proxy, user, password, px_host, px_port = parse_http_proxy_string(proxy_string)

            if (len(user) > 0 and len(password) > 0): # TODO syntax with user and pass is untested
                px_cmd = 'HTTP_PROXY_PASSWORD=' + password + ';'
                px_cmd = px_cmd + 'ProxyCommand connect -H ' + user + '@' + px_host + ':' + str(px_port) + ' %h %p'
            else:
                px_cmd = 'ProxyCommand connect -H ' + px_host + ':' + str(px_port) + ' %h %p'

        # Set -o "StrictHostKeyChecking no" -o "UserKnownHostsFile /dev/null" to work around the fact that
        # www-data user doesn't own its home directory /var/www, and so can't create an .ssh directory.
        if (have_proxy):
            args = ['/usr/bin/ssh', '-p', str(dest_port),
                '-o', 'PreferredAuthentications=publickey',
                '-o', 'StrictHostKeyChecking no',
                '-o', 'UserKnownHostsFile /dev/null',
                '-o', px_cmd,                                 # include ProxyCommand
                dest]
        else:
            args = ['/usr/bin/ssh', '-p', str(dest_port),
                '-o', 'PreferredAuthentications=publickey',
                '-o', 'StrictHostKeyChecking no',
                '-o', 'UserKnownHostsFile /dev/null',
                dest]

        p = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if ('denied' in err):
            return ":)"
        else:
            return ":("

    except:
        logger.error("ssh_status failed")
        return ":("


def how_are_you(request, host, port):
    """Attempt a connection to the given host and port, if we can :), if not :(
       Socket-to-socket connections fail in the presence of a network proxy, so use standard protocols.
       ionupdates.com gives a 403 Forbidden to wget, so fall back to socket test.
    """
    if (int(port) == 80 or int(port) == 443) and (host != 'ionupdates.com'):
        try:
            cmd = ["/usr/bin/wget", "-O", "/dev/null", "--tries", "1", "--no-check-certificate"]
            # This will suppress stdout from wget
            #cmd += ["-o", "/dev/null"]
            # Append URL
            if int(port) == 80:
                cmd += [str("http://%s:%s" % (host, str(port)))]
            elif int(port) == 443:
                cmd += [str("https://%s:%s" % (host, str(port)))]
            #logger.warn("CMD:%s" % cmd)
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = proc.communicate()
            if proc.returncode == 0:
                status = ":)"
            else:
                status = ":("
                logger.warn(stderr)
        except:
            status = ":("
            logger.warn(traceback.format_exc())
    elif int(port) == 22:
        status = ssh_status(host, port)
    else:
        # Here we revert to ignore-proxy method for lack of better option
        try:
            s = socket.create_connection((host, int(port)), 10)
            s.close()
            status = ":)"
        except Exception as complaint:
            logger.warn(complaint)
            status = ":("

    return http.HttpResponse('{"feeling":"%s"}' % status,
                             mimetype='application/javascript')


def how_am_i(request):
    """Perform a series of network status checks on the torrent server itself
    """
    result = {
        "eth_device": None,
        "route": None,
        "ip_addr": None,
    }
    try:
        stdout, stderr = script("PORT=$(route|awk '/default/{print $8}') && /sbin/ifconfig $PORT")
        for line in stdout.splitlines():
            m = re.search(r"inet addr:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
            if m:
                result["ip_addr"] = m.group(1)
            elif "UP" in line and "MTU" in line:
                result["eth_device"] = True
        stdout, stderr = script("/bin/netstat -r")
        result["route"] = "default" in stdout
    except Exception as err:
        logger.error("Exception raised during network self exam, '%s'" % err)
    return http.HttpResponse(json.dumps(result), content_type="application/json")


def fetch_remote_content(request, url):
    """Perform an HTTP GET request on the given url and forward it's result.
    This is specifically for remote requests which need to originate from this
    server rather than from the client connecting to the torrent server.
    """
    try:
        remote = urllib.urlopen(url)
        data = remote.read()
        remote.close()
    except Exception as complaint:
        logger.warn(complaint)
        data = ""
    return http.HttpResponse(data)


@staff_member_required
def network(request):
    """This calls the TSQuery, TSstaticip and TSproxy scripts to query/update
    the networking configuration.
    """
    if request.method == "POST" and "network-mode" in request.POST:
        form = NetworkConfigForm(request.POST, prefix="network")
        if form.is_valid():
            # TS* is updated by the form controller NetworkConfigForm
            form.save()
        else:
            logger.warning("User entered invalid network settings:\n%s" % str(form.errors))
        return HttpResponseRedirect("/admin/network/")
    else:
        form = NetworkConfigForm(prefix="network")

        return render_to_response("admin/network.html", RequestContext(request,
            {"network": {"mac": mac_address(), "form": form},
             "is_TsVm": is_TsVm()}))


@staff_member_required
def manage(request):
    """provide a simple interface to allow a few management actions of the Torrent Server"""

    if request.method == "POST":

        output, errors = script('sudo shutdown -P now', True)
        return render_to_response(
            "admin/manage.html",
            {"output": output, "errors": errors, "post": True},
            RequestContext(request, {}),
        )

    if request.method == "GET":

        return render_to_response(
            "admin/manage.html",
            {},
            RequestContext(request, {}),
        )


@staff_member_required
def install_log(request):
    """provide a way to output the log to the admin page"""
    log = open("/tmp/django-update").readlines()
    #log = [line.strip() for line in log]
    mime = "text/plain;charset=utf-8"
    response = http.HttpResponse(log, content_type=mime)
    return response


def install_log_text():
    try:
        return open("/tmp/django-update").readlines()
    except:
        return False


def update_locked():
    """Check to see if the update process locked"""

    lockFile = "/tmp/django-update-status"
    lockStatus = None
    if os.path.exists(lockFile):
        lockStatus = open(lockFile).readlines()
        lockStatus = lockStatus[0].strip()
    return lockStatus == "locked"


def install_lock(request):
    """provide a way to output the log to the admin page"""
    lockFile = "/tmp/django-update-status"
    mime = "text/plain;charset=utf-8"

    if os.path.exists(lockFile):
        lockStatus = open(lockFile).readlines()
        lockStatus = lockStatus[0].strip()
        response = http.HttpResponse(lockStatus, content_type=mime)
    else:
        response = http.HttpResponse(lockStatus, content_type=mime)

    return response


def tsconfig_log(request):
    """ Display tsconfig log """
    log = open("/var/log/ion/tsconfig_gui.log").readlines()
    return http.HttpResponse(log, content_type="text/plain;charset=utf-8")


def get_zip_logs(request):
    """ Make an archive of logs available to download """
    from django.core.servers.basehttp import FileWrapper
    import zipfile
    try:
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED

    zipPath = '/tmp/logs.zip'
    zipfile = zipfile.ZipFile(zipPath, mode='w', allowZip64=True)
    for afile in ['tsconfig_gui.log', 'django.log', 'celery_w1.log', 'tsconfig_debug.log']:
        fullpath = os.path.join('/var/log/ion', afile)
        if os.path.exists(fullpath):
            zipfile.write(fullpath, arcname=afile, compress_type=compression)
    zipfile.close()

    #TODO: Include the server serial number in the archive filename.
    #One possible source is /etc/torrentserver/tsconf.conf, serialnumber:XXXXXX
    archive_filename = 'ts_update_logs.zip'
    response = http.HttpResponse(FileWrapper (open(zipPath)), mimetype='application/zip')
    response['Content-Disposition'] = 'attachment; filename=%s' % archive_filename
    return response


def run_update_check(request):
    tasks.check_updates.delay()
    return http.HttpResponse()


def run_update():
    """Run the update.sh script, that will run update.sh
    'at now' needed so that when Apache restarts the update script wil not be killed
    Also check to make sure that the update process is not locked
    """
    if not update_locked():
        update_message = """Please do not start any data analysis or chip runs. The system is updating.  This may take a while, and you will see a message when it is complete."""
        Message.warn(update_message, expires="system-update-finished")
        try:
            tasks.download_updates.delay(auto_install=True)
        except Exception as err:
            logger.error("Attempting to run update but got error '%s'" % err)
            raise
        return True
    else:
        return False

def get_EULA_text():
    h = httplib2.Http()
    isValid = True
    errorMsg = None
    eula_content = None

    errorCodes = {"E001" : "The requested EULA file does not exist on the Ion Updates server",
                  "E002" : "HTTP ServerNotFoundError: {0}",
                  "E003" : "HTTP Error: {0}",
                  "E004" : "No EULA content available"}

    EULA_TEXT_URL = settings.PRODUCT_UPDATE_BASEURL + settings.EULA_TEXT_URL

    try:
        response, content = h.request(EULA_TEXT_URL)
        if response['status'] == '200':
            markDown_content = markdown.markdown(content)
            markDown_content = re.sub(r"([===*])\n", r"\1<br />", markDown_content)
            markDown_content = re.sub(r"([a-zA-Z0-9])\n([===*])", r"\1<br />\2", markDown_content)
            eula_content = markDown_content
        if response['status'] == '404':
            isValid = False
            errorMsg = errorCodes["E001"]
            logger.debug("httplib2.ServerNotFoundError: iondb.rundb.admin.py %s", errorMsg)
    except httplib2.ServerNotFoundError, err:
        isValid = False
        errorMsg = errorCodes["E002"].format(err)
        logger.debug("httplib2.ServerNotFoundError: iondb.rundb.admin.py %s", err)
    except Exception, err:
        isValid = False
        errorMsg = errorCodes["E003"].format(err)
        logger.debug("urllib2.HTTPError: iondb.rundb.admin.py %s", errorMsg)
    if not eula_content and not errorMsg:
        errorMsg = errorCodes["E004"]
        logger.debug("urllib2.HTTPError: iondb.rundb.admin.py %s", errorMsg)

    return (eula_content, isValid, errorMsg)


@staff_member_required
def update(request):
    """provide a simple interface to allow Torrent Suite to be updated"""

    if request.method == "POST":
        updateLocked = run_update()
        data = json.dumps({"lockBlocked": updateLocked})
        return http.HttpResponse(data, content_type="application/json")
    elif request.method == "GET":
        about, meta_version = findUpdates()
        config = GlobalConfig.objects.filter()[0]
        config_dict = model_to_dict(config)
        eula_content, isValid, errorMsg = get_EULA_text()

        try:
            # Disable Update Server button for some reason
            # Checking root partition for > 1GB free
            allow_update = True if files.getSpaceKB("/") > 1048576 else False
            if not allow_update:
                GlobalConfig.objects.update(ts_update_status="Insufficient disk space")
            else:
                if config.ts_update_status in "Insufficient disk space":
                    GlobalConfig.objects.update(ts_update_status="No updates")
        except:
            allow_update = True

        return render_to_response(
            "admin/update.html",
            {"about": about, "meta": meta_version,
             "show_available": config.ts_update_status not in ['No updates', 'Finished installing'],
             "eula_content": eula_content,
             "eula_error": errorMsg,
             "global_config_json": json.dumps(config_dict),
             "maintenance_mode": maintenance_action("check")['maintenance_mode'],
             "allow_update": allow_update},
            RequestContext(request, {}),
        )


@staff_member_required
def version_lock(request, enable):
    # hide repository list file /etc/apt/sources.list
    async_result = tasks.lock_ion_apt_sources.delay(enable=(enable == "enable_lock"))

    try:
        async_result.get(timeout=20)
        tasks.check_updates.delay()
    except celery.exceptions.TimeoutError as err:
        logger.warning("version_lock timed out, taking longer than 20 seconds.")

    return render_to_response("admin/update.html")


def maintenance_action(action):
    cmd = ["/opt/ion/iondb/bin/sudo_utils.py", "maintenance_mode", action]
    if action != "check":
        cmd = ["sudo"] + cmd
    try:
        p = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p.communicate()
        ret = { 'maintenance_mode':output.strip(), 'error':error }
    except Exception as err:
        ret = { 'maintenance_mode':'unknown', 'error':str(err) }
    return ret

@staff_member_required
def maintenance(request, action):
    result = maintenance_action(action)
    return http.HttpResponse(json.dumps(result), mimetype='application/json') 


def tsvm_control(request, action=''):
    if request.method == 'GET':
        log_files = []
        for logfile in ["provisioning.log", "running.log"]:
            if os.path.exists(os.path.join("/var/log/ion/", logfile)):
                log_files.append(logfile)

        ctxd = {
            'versions': tsvm.versions(),
            'status': tsvm.status(),
            'host': request.META.get('HTTP_HOST'),
            'log_files': log_files
        }
        return render_to_response("admin/tsvm_control.html", RequestContext(request, ctxd))

    elif request.method == 'POST':
        if "setup" in action:
            version = request.REQUEST.get('version', '')
            response_object = tsvm.setup(version)
        elif action in ["start", "stop", "suspend", "destroy"]:
            response_object = tsvm.ctrl(action)
        elif "status" in action:
            response_object = tsvm.status()
        elif "check_update" in action:
            stdout = subprocess.check_output(['sudo', '/opt/ion/iondb/bin/ion_check_for_new_tsvm.py'])
            response_object = json.loads(stdout)
        elif "update" in action:
            response_string = subprocess.check_output(['sudo', '/opt/ion/iondb/bin/install_new_tsvm.py'])
            response_object = json.load(response_string)
        else:
            return http.HttpResponseBadRequest('Error: unknown action type specified: %s' % action)

        return http.HttpResponse(json.dumps(response_object), content_type='application/json')


def tsvm_get_log(request, logfile):
    # Returns log file contents
    try:
        #Path specified in tsvm-include from ion-tsvm package
        filepath = os.path.join("/var/log/ion/", logfile)
        log = open(filepath).readlines()
    except IOError as e:
        log = str(e)
    response = http.HttpResponse(log, content_type="text/plain;charset=utf-8")
    return response


def ot_log(request):
    """provide a way to output the log to the admin page"""
    mime = "text/plain;charset=utf-8"
    try:
        log = open("/tmp/OTstatus").readlines()
        log = [line.strip() for line in log]
        if len(log) == 0:
            log = 'OneTouch update is running <img src="/site_media/jquery/colorbox/images/loading.gif"/>'
        else:
            log = log
        if log[-1] == "STARTED":
            log = 'OneTouch update is running <img src="/site_media/jquery/colorbox/images/loading.gif"/>'
        if log[-1] == "DONE":
            log = "OneTouch update finished! <a href='/admin'>You can leave this page</a>"
    except IOError:
        log = 'OneTouch update is waiting to start <img src="/site_media/jquery/colorbox/images/loading.gif"/>'

    response = http.HttpResponse(log, content_type=mime)
    return response


@staff_member_required
def updateOneTouch(request):
    """provide a simple interface to allow one touch updates"""

    #TODO: OT update does not provide useful log into in the case of found OTs
    if request.method == "POST":

        if not os.path.exists("/tmp/OTlock"):
            otLockFile = open("/tmp/OTlock", 'w')
            otLockFile.write("Update Started")
            otLockFile.close()
            try:
                otStatusFile = open("/tmp/OTstatus", 'w')
                otStatusFile.write("STARTED")
                otStatusFile.close()
            except:
                pass
            async = tasks.updateOneTouch.delay()
            state = 'OneTouch update is waiting to start <img src="/site_media/jquery/colorbox/images/loading.gif"/>'
        else:
            state = "Error: OneTouch update was already in progress"

        return render_to_response(
            "admin/updateOneTouch.html",
                {"post": True, "state": state},
            RequestContext(request, {}),
            )

    if request.method == "GET":

        return render_to_response(
            "admin/updateOneTouch.html",
                {},
            RequestContext(request, {}),
            )


class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('expName', 'date', 'status', 'plan')
    list_filter = ('status',)
    search_fields = ['expName']
    ordering = ('-date', 'expName', )
    actions = ['redo_from_scratch']
    raw_id_fields = ('plan', 'repResult',)

    def has_add_permission(self, request):
        return False

    # custom admin action to delete all results and reanalyze with Plan data intact
    def redo_from_scratch(self, request, queryset):
        selected = request.POST.getlist(admin.ACTION_CHECKBOX_NAME)
        return HttpResponseRedirect("/admin/experiment/exp_redo_from_scratch/?ids=%s" % (",".join(selected)))

    redo_from_scratch.short_description = "Delete and restart with Plan info"


@staff_member_required
def exp_redo_from_scratch(request):
    pks = list(request.GET.get('ids').split(','))
    exps = Experiment.objects.filter(id__in=pks)

    if request.method == "GET":

        deletable_objects = []
        for exp in exps:
            results_names = ['Results: %s' % result for result in exp.results_set.values_list('resultsName', flat=True)]
            deletable_objects.append(['Experiment: %s' % exp, results_names])

        opts = Experiment._meta
        context = {
            "app_label": opts.app_label,
            "opts": opts,
            "deletable_objects": deletable_objects,
            "url": request.get_full_path()
        }
        return render_to_response("admin/experiment_redo_from_scratch.html", RequestContext(request, context))

    if request.method == "POST":
        results = Results.objects.filter(experiment__in=pks)
        results.delete()
        # modify experiment to make crawler pick it up again
        for exp in exps:
            exp.unique = 'redo_from_scratch_' + exp.unique
            exp.ftpStatus = ''
            exp.status = 'planned'
            exp.save()
            # remove all but the original EAS
            if exp.eas_set.count() > 0:
                eas = exp.eas_set.order_by('pk')[0]
                eas.isEditable = True
                eas.isOneTimeOverride = False
                eas.save()
                exp.eas_set.exclude(id=eas.id).delete()

        admin.site._registry[Experiment].message_user(request, "%s experiments were deleted." % len(pks))
        return HttpResponseRedirect("/admin/rundb/experiment/")


class PluginResultAdmin(admin.ModelAdmin):

    def total_size(self, obj):
        if obj.size < 0:
            return "N/A"
        return filesizeformat(obj.size)
    total_size.admin_order_field = 'size'

    list_display = ('result', 'plugin', 'path', 'total_size')
    list_display_links = ('path',)
    search_fields = ('result__resultsName', 'plugin__name')
    readonly_fields = ('result', 'plugin', 'path', 'total_size')
    fields = ('result', ('plugin', ), ('path', 'total_size'), 'store',)
    ordering = ("-id", )

class PluginResultJobAdmin(admin.ModelAdmin):
    model = PluginResultJob


class PluginResultsInline(admin.StackedInline):

    def total_size(self, obj):
        if obj.size < 0:
            return "N/A"
        return filesizeformat(obj.size)
    total_size.admin_order_field = 'size'

    model = PluginResult
    verbose_name = "Plugin Result"
    extra = 0
    can_delete = True
    fields = ('plugin', ('path', 'total_size'), 'store')
    readonly_fields = ('total_size', 'plugin', 'path')
    ordering = ("-id", )


class ResultsAdmin(admin.ModelAdmin):
    list_display = ('resultsName', 'experiment', 'timeStamp', 'diskusage', 'status')
    date_hierarchy = 'timeStamp'
    search_fields = ['resultsName']
    filter_horizontal = ('projects',)
    inlines = [PluginResultsInline, ]
    readonly_fields = ('analysismetrics', 'libmetrics', 'qualitymetrics')
    raw_id_fields = ('experiment', 'eas', 'parentResult',)
    ordering = ("-id", )


class TFMetricsAdmin(admin.ModelAdmin):
    list_display = ('name', 'report')


class TemplateAdmin(admin.ModelAdmin):
    list_display = ('name', 'isofficial', 'key', 'sequence')
    ordering = ('name',)


class LocationAdmin(admin.ModelAdmin):
    list_display = ('name', 'defaultlocation')


class PluginAdmin(admin.ModelAdmin):
    list_display = ('name', 'selected', 'version', 'date', 'active', 'defaultSelected', 'path', 'url')
    list_filter = ('active', 'selected', 'defaultSelected')


class EmailAddressAdmin(admin.ModelAdmin):
    list_display = ('email', 'selected')


class GlobalConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'web_root',
                    'site_name',
                    'records_to_display',
                    'default_test_fragment_key',
                    'default_library_key',
                    'default_flow_order',
                    'plugin_output_folder',
                    'default_storage_options',
                    )
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size': '512', 'rows': 4, 'cols': 80})}
    }


class ChipAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'instrumentType', 'isActive', 'slots')
    ordering = ("name",)
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size': '512', 'rows': 4, 'cols': 80})}
    }


class dnaBarcodeAdmin(admin.ModelAdmin):
    list_display = ('name', 'id_str', 'sequence', 'adapter', 'annotation', 'score_mode', 'score_cutoff', 'type', 'length', 'floworder', 'index')
    list_filter = ('name',)


class RunTypeAdmin(admin.ModelAdmin):
    list_display = ('runType', 'description', 'application_groups')
    readonly_fields = ('applicationGroups',)

    def application_groups(self, obj):
        return ", ".join([applicationGroup.name for applicationGroup in obj.applicationGroups.all()])


class ReferenceGenomeAdmin(admin.ModelAdmin):
    list_display = ('short_name', 'name')
    # delete action doesn't remove ref files, disable it
    actions = None


class ThreePrimeadapterAdmin(admin.ModelAdmin):
    list_display = ('direction', 'chemistryType', 'name', 'sequence', 'description', 'isActive', 'isDefault')


class FlowOrderAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'flowOrder', 'isActive', 'isDefault', 'isSystem')


class LibraryKeyAdmin(admin.ModelAdmin):
    list_display = ('direction', 'name', 'sequence', 'description', 'isDefault')


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'creator', 'public', 'modified')


class PlannedExperimentQCInline(admin.TabularInline):
    model = PlannedExperiment.qcValues.through

    verbose_name = "Planned Experiment QC Thresholds"
    can_delete = False
    can_add = False

    def has_add_permission(self, request):
        return False


class PlannedExperimentAdmin(admin.ModelAdmin):
    list_display = ('planName', 'planShortID', 'date', 'planExecuted', 'isReusable', 'isSystem', 'username', 'planStatus', 'isReverseRun')
    list_filter = ('planExecuted',)
    search_fields = ['planShortID', ]
    filter_horizontal = ('projects',)
    raw_id_fields = ('latestEAS', 'parentPlan',)

    inlines = [PlannedExperimentQCInline, ]

    def has_add_permission(self, request):
        return False


class PlannedExperimentQCAdmin(admin.ModelAdmin):
    ##pass

    def has_add_permission(self, request):
        return False


class QCTypeAdmin(admin.ModelAdmin):

    def has_add_permission(self, request):
        return False


class ReportStorageAdmin(admin.ModelAdmin):
    list_display = ("name", "default", "webServerPath", "dirPath")


class FileServerAdmin(admin.ModelAdmin):
    list_display = ("name", "filesPrefix", "percentfull")


class SampleAdmin(admin.ModelAdmin):
    list_display = ("name", "displayedName", "externalId", "status")


class ExperimentAnalysisSettingsAdmin(admin.ModelAdmin):
    list_display = ("experiment", "isEditable", "isOneTimeOverride", "date", "status", "get_plan")
    list_filter = ('status',)
    search_fields = ['experiment__expName']
    ordering = ("-id", )
    raw_id_fields = ('experiment',)
    formfield_overrides = {models.CharField: {'widget': Textarea(attrs={'size': '512', 'rows': 4, 'cols': 80})}}

    def get_plan(self, obj):
        return obj.experiment.plan
    get_plan.short_description = 'Plan'


class DMFileSetAdmin(admin.ModelAdmin):
    list_display = ('type', 'include', 'exclude', 'version')


class DMFileStatAdmin(admin.ModelAdmin):
    list_display = ('result', 'dmfileset', 'diskspace', 'action_state', 'created')
    list_filter = ('action_state',)
    date_hierarchy = 'created'
    search_fields = ['result__resultsName']


class EventLogAdmin(admin.ModelAdmin):
    list_display = ('content_type', 'text', 'created')
    list_filter = ('username',)
    date_hierarchy = 'created'
    search_fields = ['text']


class AnalysisArgsAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'chipType', 'chip_default', 'active', 'sequenceKitName', 'templateKitName', 'libraryKitName', 'samplePrepKitName', "applType", "applGroup", 'isSystem', )
    ordering = ("chipType", "-chip_default", "name")
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size': '512', 'rows': 4, 'cols': 80})}
    }


class CruncherAdmin(admin.ModelAdmin):
    list_display = ('name', 'state', 'date', 'comments')
    list_filter = ('state',)


class KitInfoAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'kitType', 'templatingSize', 'flowCount', 'uid', 'categories', 'isActive', 'instrumentType', 'samplePrep_instrumentType', 'applicationType', 'chipTypes')
    ordering = ("kitType", "name",)
    list_filter = ('kitType',)

class common_CVAdmin(admin.ModelAdmin):
    list_display = ('cv_type', 'value', 'displayedValue', 'description', 'isDefault', 'isActive', 'categories', 'samplePrep_instrumentType', 'sequencing_instrumentType', 'uid')
    ordering = ("cv_type", "displayedValue",)
    list_filter = ('cv_type',)

class RigAdmin(admin.ModelAdmin):
    list_display = ('name', 'ftpserver', 'location', 'state', 'serial')
    list_filter = ('ftpserver', 'location')


class SharedServerAdmin(admin.ModelAdmin):
    list_display = ('name', 'address', 'username', 'active')


class SampleSetItemInline(admin.StackedInline):
    model = SampleSetItem
    extra = 0
    max_num = 1
    verbose_name = "Sample Set Item"
    fields = [('sample', 'dnabarcode', 'description', 'creator'), ('nucleotideType', 'pcrPlateColumn', 'pcrPlateRow'),
        ('controlType', 'gender', 'relationshipRole', 'relationshipGroup'), ('cancerType', 'cellularityPct', 'biopsyDays', 'coupleId', 'embryoId')]
    formfield_overrides = {models.CharField: {'widget': TextInput(attrs={'size': '25'})}, }


class SampleSetAdmin(admin.ModelAdmin):
    list_display = ('displayedName', 'description', 'status')
    list_filter = ('status',)
    exclude = ('libraryPrepInstrumentData', )
    inlines = [SampleSetItemInline, ]


class ApplProductAdmin(admin.ModelAdmin):
    list_display = ('productName', 'applicationGroup', 'applType', 'isDefault', 'instrumentType', 'defaultChipType', 
                    'isDefaultForInstrumentType', 'defaultLibraryKit', 'defaultTemplateKit', 'defaultIonChefPrepKit',
                    'defaultSequencingKit', 'defaultIonChefSequencingKit', 'defaultFlowCount', 
                    'isActive', 'isVisible', 'productCode')
    ordering = ("productName", "applicationGroup", "applType", "instrumentType", "defaultChipType",)
    list_filter = ('applType',)

class ApplicationGroupAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'isActive', "uid")
    list_filter = ('isActive',)
    ordering = ("name",)


class IonMeshNodeAdmin(admin.ModelAdmin):
    list_display = ('hostname', 'system_id', 'share_plans', 'share_data', 'share_monitoring')


admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Results, ResultsAdmin)
admin.site.register(Message)
admin.site.register(Location, LocationAdmin)
admin.site.register(Rig, RigAdmin)
admin.site.register(FileServer, FileServerAdmin)
admin.site.register(TFMetrics, TFMetricsAdmin)
admin.site.register(ReportStorage, ReportStorageAdmin)
admin.site.register(Cruncher, CruncherAdmin)
admin.site.register(AnalysisMetrics)
admin.site.register(LibMetrics)
admin.site.register(QualityMetrics)
admin.site.register(Template, TemplateAdmin)
admin.site.register(GlobalConfig, GlobalConfigAdmin)
admin.site.register(Plugin, PluginAdmin)
admin.site.register(PluginResult, PluginResultAdmin)
admin.site.register(PluginResultJob, PluginResultJobAdmin)
admin.site.register(EmailAddress, EmailAddressAdmin)
admin.site.register(Chip, ChipAdmin)
admin.site.register(dnaBarcode, dnaBarcodeAdmin)
admin.site.register(RunType, RunTypeAdmin)
admin.site.register(ThreePrimeadapter, ThreePrimeadapterAdmin)
admin.site.register(FlowOrder, FlowOrderAdmin)
admin.site.register(PlannedExperiment, PlannedExperimentAdmin)
admin.site.register(IonMeshNode, IonMeshNodeAdmin)
admin.site.register(Publisher)
admin.site.register(ContentUpload)
admin.site.register(Content)
admin.site.register(UserEventLog)
admin.site.register(UserProfile)
#ref genome
admin.site.register(ReferenceGenome, ReferenceGenomeAdmin)

admin.site.register(Project, ProjectAdmin)

admin.site.register(KitInfo, KitInfoAdmin)
admin.site.register(KitPart)
admin.site.register(LibraryKey, LibraryKeyAdmin)

admin.site.register(QCType, QCTypeAdmin)
admin.site.register(ApplProduct, ApplProductAdmin)

admin.site.register(PlannedExperimentQC, PlannedExperimentQCAdmin)

admin.site.register(Sample, SampleAdmin)
admin.site.register(ExperimentAnalysisSettings, ExperimentAnalysisSettingsAdmin)
admin.site.register(DMFileSet, DMFileSetAdmin)
admin.site.register(DMFileStat, DMFileStatAdmin)
admin.site.register(EventLog, EventLogAdmin)
admin.site.register(RemoteAccount)
admin.site.register(FileMonitor)
admin.site.register(SupportUpload)
admin.site.register(NewsPost)
admin.site.register(AnalysisArgs, AnalysisArgsAdmin)
admin.site.register(SharedServer, SharedServerAdmin)
admin.site.register(SampleSet, SampleSetAdmin)
admin.site.register(common_CV, common_CVAdmin)
admin.site.register(ApplicationGroup, ApplicationGroupAdmin)

# Add sessions to admin
class SessionAdmin(admin.ModelAdmin):

    def _session_data(self, obj):
        return obj.get_decoded()
    list_display = ['session_key', '_session_data', 'expire_date']
admin.site.register(Session, SessionAdmin)

class PlanSessionAdmin(admin.ModelAdmin):
    list_display = ['session_key', 'plan_key', 'expire_date']
admin.site.register(PlanSession, PlanSessionAdmin)
