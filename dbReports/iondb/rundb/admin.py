# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import *
from iondb.rundb import tasks
from iondb.rundb import forms
from iondb.rundb import views
from django.contrib import admin
from django.forms import TextInput, Textarea

from django.template import RequestContext
from django.template.defaultfilters import filesizeformat
from django.shortcuts import render_to_response
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

from ion.utils.TSversion import findVersions

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


def how_are_you(request, host, port):
    """Open a socket to the given host and port, if we can :), if not :(
    """
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
        form = forms.NetworkConfigForm(request.POST, prefix="network")
        if form.is_valid():
            # TS* is updated by the form controller forms.NetworkConfigForm
            form.save()
        else:
            logger.warning("User entered invalid network settings:\n%s" % str(form.errors))
    else:
        form = forms.NetworkConfigForm(prefix="network")

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
        Message.warn(update_message, expires="startup")
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
        about, meta_version = findVersions()
        config = GlobalConfig.get()
        from iondb.rundb.api import GlobalConfigResource
        resource = GlobalConfigResource()
        bundle = Bundle(config)
        serialized_config = resource.serialize(None,
                                               resource.full_dehydrate(bundle),
                                               "application/json")
        return render_to_response(
            "admin/update.html",
            {"about": about, "meta": meta_version,
             "global_config": serialized_config},
            RequestContext(request, {}),
        )


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
    list_display = ('expName', 'date')
    search_fields = ['expName' ]
    ordering = ('-date', 'expName', )

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
    fields = ('result', ('plugin', 'state'), ('starttime', 'endtime', 'duration'), ('path','total_size'), 'store')
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
    list_display = ('resultsName','experiment','timeStamp')
    date_hierarchy = 'timeStamp'
    search_fields = ['resultsName']
    filter_horizontal = ('projects',)
    inlines = [ PluginResultsInline, ]
    ordering = ( "-id", )

class TFMetricsAdmin(admin.ModelAdmin):
    list_display = ('name', 'report')


class TemplateAdmin(admin.ModelAdmin):
    list_display = ('name',)


class LocationAdmin(admin.ModelAdmin):
    list_display = ('name','defaultlocation')


class BackupAdmin(admin.ModelAdmin):
    list_display = ('experiment',)


class BackupConfigAdmin(admin.ModelAdmin):
    list_display = ('backupDirectory',)


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
                    'default_plugin_script',
                    'default_storage_options',
                    )
    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'size':'512','rows':4,'cols':80})}
    }


class ChipAdmin(admin.ModelAdmin):
    list_display = ('name','description','slots')
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
    list_display = ('direction', 'name','sequence','description', 'isDefault')


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
    list_display = ('planName','planShortID','date','isReverseRun','planExecuted')
    list_filter = ('planExecuted',)
    search_fields = ['planShortID',]
    filter_horizontal = ('projects',)

    inlines = [PlannedExperimentQCInline,]

class PlannedExperimentQCAdmin(admin.ModelAdmin):
    ##pass

    def has_add_permission(self, request):
        return False

class QCTypeAdmin(admin.ModelAdmin):
        
    def has_add_permission(self, request):
        return False

class DM_Reports(admin.ModelAdmin):
	list_display = ("location","pruneLevel")
    
class DM_PruneGroup(admin.ModelAdmin):
	list_display = ("name","pk","ruleNums","editable")
    
class DM_PruneRule(admin.ModelAdmin):
	list_display = ("id","rule")

class ReportStorageAdmin(admin.ModelAdmin):
    list_display = ("name", "default")

#logger.exception("Registering admin site pages")

admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Results, ResultsAdmin)
admin.site.register(Message)
admin.site.register(Location,LocationAdmin)
admin.site.register(Rig)
admin.site.register(FileServer)
admin.site.register(TFMetrics, TFMetricsAdmin)
admin.site.register(ReportStorage,ReportStorageAdmin)
admin.site.register(RunScript)
admin.site.register(Cruncher)
admin.site.register(AnalysisMetrics)
admin.site.register(LibMetrics)
admin.site.register(QualityMetrics)
admin.site.register(Template)
admin.site.register(Backup)
admin.site.register(BackupConfig)
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
admin.site.register(dm_reports,DM_Reports)
admin.site.register(dm_prune_group,DM_PruneGroup)
admin.site.register(dm_prune_field,DM_PruneRule)
#ref genome
admin.site.register(ReferenceGenome,ReferenceGenomeAdmin)

admin.site.register(Project,ProjectAdmin)

admin.site.register(KitInfo)
admin.site.register(KitPart)
admin.site.register(LibraryKey, LibraryKeyAdmin)

admin.site.register(QCType, QCTypeAdmin)
admin.site.register(ApplProduct)

admin.site.register(PlannedExperimentQC, PlannedExperimentQCAdmin)


# Add sessions to admin
class SessionAdmin(admin.ModelAdmin):
    def _session_data(self, obj):
        return obj.get_decoded()
    list_display = ['session_key', '_session_data', 'expire_date']
admin.site.register(Session, SessionAdmin)
