# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

from iondb.rundb.models import *
from iondb.rundb import tasks
from iondb.rundb import tsvm
from iondb.rundb.forms import NetworkConfigForm
from iondb.utils import files
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

from ion.utils.TSversion import findUpdates

logger = logging.getLogger(__name__)


def script(script_text, shell_bool = True):
    """run system commands"""
    p = Popen(args=script_text, shell=shell_bool, stdout=PIPE, stdin=PIPE)
    output, errors = p.communicate()
    return output, errors


def mac_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', 'eth0'))
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
            status  = ":("

    return http.HttpResponse('{"feeling":"%s"}' % status,
                             mimetype='application/javascript')


def how_am_i(request):
    """Perform a series of network status checks on the torrent server itself
    """
    result = {
        "eth0": None,
        "route": None,
        "ip_addr": None,
    }
    try:
        stdout, stderr = script("/sbin/ifconfig eth0")
        for line in stdout.splitlines():
            m = re.search(r"inet addr:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
            if m:
                result["ip_addr"] = m.group(1)
            elif "UP" in line and "MTU" in line:
                result["eth0"] = True
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
    else:
        form = NetworkConfigForm(prefix="network")

    return render_to_response("admin/network.html", RequestContext(request,
        {"network": {"mac": mac_address(), "form": form} }))


@staff_member_required
def manage(request):
    """provide a simple interface to allow a few management actions of the Torrent Server"""

    if request.method=="POST":

        output, errors = script('sudo shutdown -P now',True)
        return render_to_response(
            "admin/manage.html",
            {"output": output, "errors" : errors, "post": True},
            RequestContext(request, {}),
        )

    if request.method=="GET":

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
    ''' Make an archive of logs available to download '''
    from django.core.servers.basehttp import FileWrapper
    import zipfile
    try:
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED

    zipPath = '/tmp/logs.zip'
    zipfile = zipfile.ZipFile(zipPath, mode='w', allowZip64=True)
    files = ['tsconfig_gui.log','django.log','celery_w1.log']
    for afile in files:
        fullpath = os.path.join('/var/log/ion',afile)
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


@staff_member_required
def update(request):
    """provide a simple interface to allow Torrent Suite to be updated"""

    if request.method=="POST":
        updateLocked = run_update()
        data = json.dumps({"lockBlocked" : updateLocked })
        return http.HttpResponse(data, content_type="application/json")
    elif request.method=="GET":
        about, meta_version = findUpdates()
        # don't use cached method here, update page needs current info
        config = GlobalConfig.objects.filter()[0]
        config_dict = model_to_dict(config)
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
             "global_config_json": json.dumps(config_dict),
             "allow_update": allow_update},
            RequestContext(request, {}),
        )


@staff_member_required
def version_lock(request, enable):

    if enable == "enable_lock":
        # hide repository list file /etc/apt/sources.list
        async_result = tasks.lock_ion_apt_sources.delay(enable=True)
    else:
        # restore repository list file /etc/apt/sources.list
        async_result = tasks.lock_ion_apt_sources.delay(enable=False)

    try:
        async_result.get(timeout=20)
        tasks.check_updates.delay()
    except celery.exceptions.TimeoutError as err:
        logger.warning("version_lock timed out, taking longer than 20 seconds.")

    return render_to_response("admin/update.html")


def tsvm_control(request, action=''):

    if request.method == 'GET':
        status = tsvm.status()
        versions = tsvm.versions()
        log_files = []
        for logfile in ["provisioning.log", "running.log"]:
            if os.path.exists(os.path.join("/var/log/ion/",logfile)):
                log_files.append(logfile)

        ctxd = {
            'versions': versions.split() if len(versions) > 0 else '',
            'status': status,
            'host': request.META.get('HTTP_HOST'),
            'log_files': log_files
        }
        return render_to_response("admin/tsvm_control.html", RequestContext(request, ctxd))
    
    elif request.method == 'POST':
        if "setup" in action:
            version = request.REQUEST.get('version','')
            response_object = tsvm.setup(version)
        elif action in ["start","stop","suspend","destroy"]:
            response_object = tsvm.ctrl(action)
        elif "status" in action:
            response_object = tsvm.status()
        elif "check_update" in action:
            async_result = tsvm.check_for_new_tsvm.delay()
            response_object = async_result.get()
        elif "update" in action:
            async_result = tsvm.install_new_tsvm.delay()
            response_object = async_result.get()
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
    if request.method=="POST":

        if not os.path.exists("/tmp/OTlock"):
            otLockFile = open("/tmp/OTlock",'w')
            otLockFile.write("Update Started")
            otLockFile.close()
            try:
                otStatusFile = open("/tmp/OTstatus",'w')
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
                {"post": True , "state" : state },
            RequestContext(request, {}),
            )

    if request.method=="GET":

        return render_to_response(
            "admin/updateOneTouch.html",
                {},
            RequestContext(request, {}),
            )


class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('expName', 'date', 'status', 'plan')
    list_filter = ('status',)
    search_fields = ['expName' ]
    ordering = ('-date', 'expName', )
    actions = ['redo_from_scratch']
    raw_id_fields = ('plan','repResult',)

    # custom admin action to delete all results and reanalyze with Plan data intact
    def redo_from_scratch(self, request, queryset):
        selected = request.POST.getlist(admin.ACTION_CHECKBOX_NAME)
        return HttpResponseRedirect("/admin/experiment/exp_redo_from_scratch/?ids=%s" % (",".join(selected)))

    redo_from_scratch.short_description = "Delete and restart with Plan info"

@staff_member_required
def exp_redo_from_scratch(request):
    pks = list( request.GET.get('ids').split(',') )
    exps = Experiment.objects.filter(id__in=pks)

    if request.method=="GET":

        deletable_objects = []
        for exp in exps:
            results_names = ['Results: %s' % result for result in exp.results_set.values_list('resultsName',flat=True)]
            deletable_objects.append( ['Experiment: %s' % exp, results_names ] )

        opts = Experiment._meta
        context = {
            "app_label": opts.app_label,
            "opts": opts,
            "deletable_objects": deletable_objects,
            "url": request.get_full_path()
        }
        return render_to_response("admin/experiment_redo_from_scratch.html", RequestContext(request, context))

    if request.method=="POST":
        results = Results.objects.filter(experiment__in=pks)
        results.delete()
        # modify experiment to make crawler pick it up again
        for exp in exps:
            exp.unique = exp.plan.planGUID
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
    def total_size(self,obj):
        if obj.size < 0:
            return "N/A"
        return filesizeformat(obj.size)
    total_size.admin_order_field = 'size'

    list_display = ('result', 'plugin', 'state', 'path', 'duration', 'total_size')
    list_display_links = ('path',)
    list_filter = ('state',)
    search_fields = ('result', 'plugin')
    readonly_fields = ('result', 'plugin', 'duration', 'path', 'total_size')
    fields = ('result', ('plugin', 'state'), ('starttime', 'endtime', 'duration'), ('path','total_size'), 'store',)
    ordering = ( "-id", )

class PluginResultsInline(admin.StackedInline):
    def total_size(self,obj):
        if obj.size < 0:
            return "N/A"
        return filesizeformat(obj.size)
    total_size.admin_order_field = 'size'

    model = PluginResult
    verbose_name = "Plugin Result"
    extra = 0
    can_delete = True
    fields = (('plugin', 'state'), ('starttime', 'endtime', 'duration'), ('path', 'total_size'), 'store')
    readonly_fields = ('duration', 'total_size', 'plugin', 'path')
    radio_fields = {'state': admin.HORIZONTAL}
    ordering = ( "endtime", "-id", )

class ResultsAdmin(admin.ModelAdmin):
    list_display = ('resultsName','experiment','timeStamp','diskusage','status')
    date_hierarchy = 'timeStamp'
    search_fields = ['resultsName']
    filter_horizontal = ('projects',)
    inlines = [ PluginResultsInline, ]
    readonly_fields = ('analysismetrics', 'libmetrics', 'qualitymetrics')
    raw_id_fields = ('experiment','eas',)
    ordering = ( "-id", )

class TFMetricsAdmin(admin.ModelAdmin):
    list_display = ('name', 'report')


class TemplateAdmin(admin.ModelAdmin):
    list_display = ('name',)


class LocationAdmin(admin.ModelAdmin):
    list_display = ('name','defaultlocation')


class BackupAdmin(admin.ModelAdmin):
    list_display = ('experiment','backupDate','backupPath')
    search_fields = ['backupName' ]
    ordering = ( "-id", )

    def has_add_permission(self, request):
        return False

class BackupConfigAdmin(admin.ModelAdmin):
    list_display = ('backup_directory',)

    def has_add_permission(self, request):
        return False


class PluginAdmin(admin.ModelAdmin):
    list_display = ('name','selected','version','date','active','path','url')
    list_filter = ('active','selected')

class EmailAddressAdmin(admin.ModelAdmin):
    list_display = ('email','selected')


class GlobalConfigAdmin(admin.ModelAdmin):
    list_display = ('name','web_root',
                    'site_name',
                    'plugin_folder',
                    'records_to_display',
                    'default_test_fragment_key',
                    'default_library_key',
                    'default_flow_order',
                    'plugin_output_folder',
                    'default_storage_options',
                    )
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size':'512','rows':4,'cols':80})}
    }


class ChipAdmin(admin.ModelAdmin):
    list_display = ('name','description','instrumentType', 'isActive', 'slots')
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size':'512','rows':4,'cols':80})}
    }


class dnaBarcodeAdmin(admin.ModelAdmin):
    list_display = ('name','id_str','sequence','adapter','annotation','score_mode','score_cutoff','type','length','floworder','index')
    list_filter  = ('name',)


class RunTypeAdmin(admin.ModelAdmin):
    list_display = ('runType','description')


class ReferenceGenomeAdmin(admin.ModelAdmin):
    list_display = ('short_name','name')


class ThreePrimeadapterAdmin(admin.ModelAdmin):
    list_display = ('direction', 'chemistryType', 'name','sequence','description', 'isDefault')


class LibraryKeyAdmin(admin.ModelAdmin):
    list_display = ('direction', 'name','sequence','description', 'isDefault')

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
    list_display = ('planName','planShortID','date', 'planExecuted', 'isReusable','isSystem', 'username', 'planStatus', 'isReverseRun')
    list_filter = ('planExecuted',)
    search_fields = ['planShortID',]
    filter_horizontal = ('projects',)
    raw_id_fields = ('latestEAS','parentPlan',)

    inlines = [PlannedExperimentQCInline,]

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
    list_display = ("experiment", "isEditable", "isOneTimeOverride", "date", "status")
    list_filter = ('status',)
    search_fields = ['experiment__expName' ]
    ordering = ( "-id", )
    raw_id_fields = ('experiment',)
    formfield_overrides = { models.CharField: {'widget': Textarea(attrs={'size':'512','rows':4,'cols':80})} }

class DMFileSetAdmin(admin.ModelAdmin):
    list_display = ('type','include','exclude','version')

class DMFileStatAdmin(admin.ModelAdmin):
    list_display = ('result','dmfileset', 'diskspace', 'action_state', 'created')
    list_filter = ('action_state',)
    date_hierarchy = 'created'
    search_fields = ['result__resultsName']

class EventLogAdmin(admin.ModelAdmin):
    list_display = ('content_type','text','created')
    list_filter = ('username',)
    date_hierarchy = 'created'
    search_fields = ['text']

class AnalysisArgsAdmin(admin.ModelAdmin):
    list_display = ('name','chipType','chip_default', 'sequenceKitName', 'templateKitName', 'libraryKitName', 'samplePrepKitName')
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size':'512','rows':4,'cols':80})}
    }

class CruncherAdmin(admin.ModelAdmin):
    list_display = ('name','state','date','comments')
    list_filter = ('state',)

class KitInfoAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'kitType', 'instrumentType','isActive')
    list_filter = ('kitType',)
    
class RigAdmin(admin.ModelAdmin):
    list_display = ('name', 'ftpserver', 'location', 'state', 'serial')
    list_filter = ('ftpserver', 'location')

class SharedServerAdmin(admin.ModelAdmin):
    list_display = ('name', 'address', 'username', 'active')

admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Results, ResultsAdmin)
admin.site.register(Message)
admin.site.register(Location,LocationAdmin)
admin.site.register(Rig, RigAdmin)
admin.site.register(FileServer,FileServerAdmin)
admin.site.register(TFMetrics, TFMetricsAdmin)
admin.site.register(ReportStorage,ReportStorageAdmin)
admin.site.register(Cruncher,CruncherAdmin)
admin.site.register(AnalysisMetrics)
admin.site.register(LibMetrics)
admin.site.register(QualityMetrics)
admin.site.register(Template)
admin.site.register(Backup, BackupAdmin)
admin.site.register(BackupConfig, BackupConfigAdmin)
admin.site.register(GlobalConfig, GlobalConfigAdmin)
admin.site.register(Plugin, PluginAdmin)
admin.site.register(PluginResult, PluginResultAdmin)
admin.site.register(EmailAddress, EmailAddressAdmin)
admin.site.register(Chip,ChipAdmin)
admin.site.register(dnaBarcode,dnaBarcodeAdmin)
admin.site.register(RunType,RunTypeAdmin)
admin.site.register(ThreePrimeadapter,ThreePrimeadapterAdmin)
admin.site.register(PlannedExperiment,PlannedExperimentAdmin)
admin.site.register(Publisher)
admin.site.register(ContentUpload)
admin.site.register(Content)
admin.site.register(UserEventLog)
admin.site.register(UserProfile)
admin.site.register(VariantFrequencies)
#ref genome
admin.site.register(ReferenceGenome,ReferenceGenomeAdmin)

admin.site.register(Project,ProjectAdmin)

admin.site.register(KitInfo, KitInfoAdmin)
admin.site.register(KitPart)
admin.site.register(LibraryKey, LibraryKeyAdmin)

admin.site.register(QCType, QCTypeAdmin)
admin.site.register(ApplProduct)

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
admin.site.register(AnalysisArgs,AnalysisArgsAdmin)
admin.site.register(SharedServer,SharedServerAdmin)

# Add sessions to admin
class SessionAdmin(admin.ModelAdmin):
    def _session_data(self, obj):
        return obj.get_decoded()
    list_display = ['session_key', '_session_data', 'expire_date']
admin.site.register(Session, SessionAdmin)
