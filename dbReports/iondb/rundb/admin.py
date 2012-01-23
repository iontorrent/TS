# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import *
from iondb.rundb import tasks
from iondb.rundb import forms
from django.contrib import admin

from django.template import RequestContext
from django.shortcuts import render_to_response
from django.contrib.admin.views.decorators import staff_member_required

from django import http

from subprocess import Popen, PIPE

import os
import socket
import fcntl
import struct

def script(script_text, shell_bool = True):
    """run system commands"""
    p = Popen(args=script_text, shell=shell_bool, stdout=PIPE, stdin=PIPE)
    output, errors = p.communicate()
    return output, errors


def mac_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', 'eth0'))
    return ''.join(['%02x:' % ord(char) for char in info[18:24]])[:-1]


@staff_member_required
def network(request):
    """This calls the TSQuery, TSstaticip and TSproxy scripts to query/update
    the networking configuration.
    """
    if request.method == "POST" and "network-mode" in request.POST:
        form = forms.NetworkConfigForm(request.POST, prefix="network")
        if form.is_valid():
            form.save()
        else:
            logger.warning("User entered invalid network settings:\n%s" % str(form.errors))
    else:
        form = forms.NetworkConfigForm(prefix="network")

    return render_to_response("admin/network.html", {"network": {
        "mac": mac_address(),
        "form": form,
        }
    })



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
    
manage = staff_member_required(manage)

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

    if os.path.exists(lockFile):
        lockStatus = open(lockFile).readlines()
        lockStatus = lockStatus[0].strip()
        if lockStatus == "locked":
            return True

    return False

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

install_log = staff_member_required(install_log)

def run_update():
    """Run the update.sh script, that will run update.sh
    'at now' needed so that when Apache restarts the update script wil not be killed
    Also check to make sure that the update process is not locked
    """
    if not update_locked():
        output, errors = script('sudo at now < /opt/ion/iondb/bin/update.sh > /dev/null ',True)
        return True
    else:
        return False

def update(request):
    """provide a simple interface to allow Torrent Suite to be updated"""
    
    if request.method=="POST":
    
        updateLocked = run_update()
        
        return render_to_response(
            "admin/update.html",
            {"post": True, "lockBlocked" : updateLocked },
            RequestContext(request, {}),
        )
    
    if request.method=="GET":
    
        return render_to_response(
            "admin/update.html",
            {},
            RequestContext(request, {}),
        )
    
update = staff_member_required(update)

def ot_log(request):
    """provide a way to output the log to the admin page"""
    mime = "text/plain;charset=utf-8"
    try:
        log = open("/tmp/OTstatus").readlines()
        log = [line.strip() for line in log]
        if len(log) == 0:
            log = "Waiting for update to run...."
    except:
        log = "OneTouch update is no longer running, <a href='/admin'>you can leave this page.</a>"

    response = http.HttpResponse(log, content_type=mime)
    return response

def updateOneTouch(request):
    """provide a simple interface to allow one touch updates"""

    if request.method=="POST":

        if not os.path.exists("/tmp/OTlock"):
            otLockFile = open("/tmp/OTlock",'w')
            otLockFile.write("Update Started")
            async = tasks.updateOneTouch.delay()
            otLockFile.close()
            state = "OneTouch update has started"
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

updateOneTouch = staff_member_required(updateOneTouch)

class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('expName', 'date')

class ResultsAdmin(admin.ModelAdmin):
    list_display = ('resultsName', 'experiment','timeStamp')
    search_fields = ['resultsName' ]

class TFMetricsAdmin(admin.ModelAdmin):
    list_display = ('name', 'report')

class TemplateAdmin(admin.ModelAdmin):
    list_display = ('name',)

class LocationAdmin(admin.ModelAdmin):
    list_display = ('name',)
    
class BackupAdmin(admin.ModelAdmin):
    list_display = ('experiment',)

class BackupConfigAdmin(admin.ModelAdmin):
    list_display = ('backupDirectory',)

class PluginAdmin(admin.ModelAdmin):
    list_display = ('name','selected','version','date','active','path')

class PluginResultAdmin(admin.ModelAdmin):
    list_display = ('result', 'plugin', 'state', 'store')

class EmailAddressAdmin(admin.ModelAdmin):
    list_display = ('email','selected')

class GlobalConfigAdmin(admin.ModelAdmin):
    list_display = ('name','web_root',
                    'site_name',
                    'plugin_folder',
                    'default_command_line',
                    'records_to_display',
                    'default_test_fragment_key',
                    'default_library_key',
                    'default_flow_order',
                    'plugin_output_folder',
                    'default_plugin_script',
                    'default_storage_options',
                    )
class ChipAdmin(admin.ModelAdmin):
    list_display = ('name','slots','args')

class dnaBarcodeAdmin(admin.ModelAdmin):
    list_display = ('name','id_str','sequence','adapter','annotation','score_mode','score_cutoff','type','length','floworder','index')
    list_filter  = ('name',)

class RunTypeAdmin(admin.ModelAdmin):
    list_display = ('runType','description')

class ReferenceGenomeAdmin(admin.ModelAdmin):
    list_display = ('short_name','name')
    
class ThreePrimeadapterAdmin(admin.ModelAdmin):
    list_display = ('name','sequence','description')


admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Results, ResultsAdmin)
admin.site.register(Location,LocationAdmin)
admin.site.register(Rig)
admin.site.register(FileServer)
admin.site.register(TFMetrics, TFMetricsAdmin)
admin.site.register(ReportStorage)
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
admin.site.register(PlannedExperiment)
admin.site.register(Publisher)
admin.site.register(ContentUpload)
admin.site.register(Content)
admin.site.register(UserEventLog)
admin.site.register(UserProfile)
admin.site.register(SequencingKit)
admin.site.register(LibraryKit)
admin.site.register(VariantFrequencies)
#ref genome
admin.site.register(ReferenceGenome,ReferenceGenomeAdmin)
