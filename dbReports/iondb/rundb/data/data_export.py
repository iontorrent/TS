# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
from celery.task import task, periodic_task
from celery.utils.log import get_task_logger
from celery.result import AsyncResult
import celery.states
from datetime import timedelta
from iondb.rundb import models
import errno
import time
import hashlib
import json
import os
import os.path
import requests
from django import forms
from django.conf import settings
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
import boto
import boto.s3.key

from iondb.utils import makePDF
from ion.utils import makeCSA
from iondb.rundb.api import SupportUploadResource

logger = get_task_logger(__name__)

@login_required
def list_export_uploads(request, tag):

    uploads = models.FileMonitor.objects.filter(tags__contains="upload").order_by('-created')
    if tag:
        uploads = uploads.filter(tags_contains=tag)
    ctx = {
        'uploads': uploads
    }
    return render(request, "rundb/data/data_export.html", ctx)


def md5_stats_file(path):
    with open(path, 'rb', 8192) as fp:
        digest_hex, diges_64, size = boto.s3.key.compute_md5(fp)
    return digest_hex, diges_64, size

@task
def export_upload_file(monitor_id):
    monitor = models.FileMonitor.objects.get(pk=monitor_id)
    full = monitor.full_path()
    monitor.status = "Checking"
    monitor.save()
    if not os.path.exists(full):
        logger.error("OS Error in file uploader")
        monitor.status = "Error: file does not exist"
        monitor.save()
        return

    digest_hex, diges_64, size = md5_stats_file(full)
    monitor.size = size
    monitor.md5sum = digest_hex
    monitor.url = "{0}:{1}".format(monitor.name, monitor.md5sum)

    monitor.status = "Connecting"
    monitor.save()

    try:
        con = boto.connect_s3(settings.AWS_ACCESS_KEY, settings.AWS_SECRET_KEY)
        bucket = con.get_bucket(settings.AWS_BUCKET_NAME)
        key = bucket.get_key(monitor.url)
        if key is not None:
            monitor.status = "Complete"
            monitor.save()
            return
        key = bucket.new_key(monitor.url)
    except Exception as err:
        logger.exception("Connecting error")
        monitor.status = "Error: {0}".format(err)
        monitor.save()
        return

    monitor.status = "Uploading"
    monitor.save()

    # Rewrite this into a class or a callable object
    #last_time = time.time()
    def get_progress(current, total):
        #now = time.time()
        #if now - last_time >= 0.5:
        monitor.progress = current
        monitor.save()
            #last_time = now

    try:
        key.set_contents_from_filename(full, cb=get_progress, num_cb=1000,
            md5=(digest_hex, diges_64))
    except Exception as err:
        logger.exception("Uploading error")
        monitor.status = "Error: Uploading {0}".format(err)[:60]
        monitor.save()
        return

    monitor.progress = monitor.size
    monitor.status = "Complete"
    monitor.save()


@login_required
def export_upload_report(request):
    try:
        report_pk = int(request.POST.get("report"))
    except ValueError:
        raise Http404("'{0}' is not a valid report ID".format(report_pk))
    path = request.POST.get("file_path")
    report = get_object_or_404(models.Results, pk=report_pk)
    root = report.get_report_dir()
    full_path = os.path.join(root, path)
    if not os.path.exists(full_path):
        raise Http404("'{0}' does not exist as a file in report {1}".format(path, report_pk))
    tag = "report/{0}/".format(report_pk)
    monitor = models.FileMonitor(
        local_dir = os.path.dirname(full_path),
        name = os.path.basename(full_path),
        tags = "upload,"+tag,
        status = "Queued"
    )

    monitor.save()
    result = export_upload_file.delay(monitor.id)
    monitor.celery_task_id = result.task_id
    monitor.save()

    return redirect("list_export_uploads", tag="")


class SupportUploadForm(forms.Form):

    contact_email = forms.EmailField(required=True)
    description = forms.CharField(required=True)
    result = forms.IntegerField(required=True)


@task
def filemonitor_errback(task_id, monitor_pk):
    try:
        monitor = models.FileMonitor.objects.get(pk=monitor_pk)
    except:
        logger.exception("Monitor error callback failed for pk={0}".format(monitor_pk))
        return
    monitor.status = "Error"
    monitor.save()


@task
def generate_csa(result_pk, monitor_pk=None):
    result = models.Results.objects.get(pk=result_pk)
    report_dir = result.get_report_dir()
    raw_data_dir = result.experiment.expDir

    try:
        monitor = models.FileMonitor.objects.get(pk=monitor_pk)
    except models.FileMonitor.DoesNotExist:
        monitor = models.FileMonitor()
        monitor.tags = "generate_csa"

    csa_file_name = "csa_{0:04d}.zip".format(int(result_pk))
    monitor.status = "Generating"
    monitor.local_dir = report_dir
    monitor.name = csa_file_name
    monitor.save()

    # Generate report PDF file.
    # This will create a file named report.pdf in results directory
    makePDF.write_report_pdf(result_pk)
    csa_path = makeCSA.makeCSA(report_dir, raw_data_dir, monitor.name)

    digest_hex, digest_64, size = md5_stats_file(csa_path)
    monitor.md5sum = digest_hex
    monitor.size = size
    monitor.status = "Generated"
    monitor.save()


def get_ts_info():
    path = '/etc/torrentserver/tsconf.conf'
    d = dict()
    if os.path.exists(path):
        for l in open(path):
            row = map(str.strip, l.split(':', 1))
            if len(row) == 2:
                d[row[0]] = row[1]
    return d


def make_auth_header(account):
    return {'Authorization': 'Bearer {0}'.format(account.access_token)}


@task
def is_authenticated(account_pk):
    account = models.RemoteAccount.objects.get(pk=account_pk)
    if account.has_access():
        url = settings.SUPPORT_AUTH_URL
        auth_header = make_auth_header(account)
        info = get_ts_info()
        params = {
            'version': info.get('version', 'Version Missing'),
            'serial_number': info.get('serialnumber', 'Serial Missing')
        }
        response = requests.get(url, params=params, headers=auth_header)
        if response.ok:
            return True
        else:
            logger.error("Authentication failure at {0}: {1}".format(url, response))
    return False


@task
def check_authentication(support_upload_pk):
    upload = models.SupportUpload.objects.get(pk=support_upload_pk)
    upload.local_status = "Authenticating"
    upload.save()
    if is_authenticated(upload.account_id):
        upload.local_status = "Authenticated"
        upload.save()
        return True
    else:
        upload.local_status = "Access Denied"
        upload.save()
        return False


@periodic_task(run_every=timedelta(hours=12), queue="periodic")
def poll_support_site():
    """This function checks to determine whether or not the customer support
    back-end site is accessible.  The purpose of this check is to allow the
    user to manually initiate an upload of the CSA to the support server;
    however, we don't want to present the UI to do this on servers which are
    offline where it won't work anyway.
    This does not initiate or enable the transfer of any data except by manual
    user action, on a case by case basis, from the report page.
    """
    config = models.GlobalConfig.get()
    account, created = models.RemoteAccount.objects.get_or_create(remote_resource="support.iontorrent", defaults={"account_label": "Ion Torrent Support"})
    url = settings.SUPPORT_AUTH_URL
    auth_header = make_auth_header(account)
    info = get_ts_info()
    params = {
        'version': info.get('version', 'Version Missing'),
        'serial_number': info.get('serialnumber', 'Serial Missing'),
        'poll': True,
    }
    try:
        response = requests.get(url, params=params, headers=auth_header, verify=False)
    except requests.exceptions.ConnectionError:
        pass
    else:
        if response.ok:
            config.enable_support_upload = True
            config.save()
            return True
    # Either there was an exception or the response was not OK
    config.enable_support_upload = False
    config.save()
    return False


@task
def upload_to_support(support_upload_pk):
    upload = models.SupportUpload.objects.select_related('file', 'account').get(pk=support_upload_pk)
    upload.local_status = "Uploading"
    upload.save()

    url = settings.SUPPORT_UPLOAD_URL
    auth_header = {'Authorization': 'Bearer ' + upload.account.access_token}
    info = get_ts_info()
    form_data = {
        'contact_email': upload.contact_email,
        'description': upload.description,
        'version': info.get('version', 'Version Missing'),
        'serial_number': info.get('serialnumber', 'Serial Missing')
    }
    path = upload.file.full_path()
    files = {'file': open(path, 'rb')}

    try:
        response = requests.post(url, data=form_data, files=files, headers=auth_header, verify=False)
    except Exception as err:
        upload.local_status = "Error"
        upload.local_message = str(err)
    else:
        if response.ok:
            try:
                tick = response.json()
            except ValueError:
                tick = {}
            upload.local_status = "Complete"
            upload.ticket_id = tick.get("ticket_id", "None")
            upload.ticket_status = tick.get("ticket_status", "Remote Error")
            upload.ticket_message = tick.get("ticket_message", "There was an error in the support server.  Your Torrent Server is working fine, and you should contact your support representative.")
        else:
            upload.local_status = "Error"
            upload.local_message = response.reason
    finally:
        upload.save()


@task(max_retries=120)
def check_and_upload(support_upload_pk, auth_task, gen_task):
    auth = AsyncResult(auth_task)
    gen_csa = AsyncResult(gen_task)
    if not (auth.ready() and gen_csa.ready()):
        check_and_upload.retry(countdown=1)
    elif auth.status == "SUCCESS" and gen_csa.status == "SUCCESS":
        if auth.result:
            upload_to_support(support_upload_pk)
    else:
        return


@login_required
@require_POST
def report_support_upload(request):
    form = SupportUploadForm(request.POST)
    # check for existing support upload
    data = {
        "created": False,
    }
    account = models.RemoteAccount.objects.get(remote_resource="support.iontorrent")
    if not account.has_access():
        data["error"] = "invalid_auth"
    if not form.is_valid():
        data["error"] = "invalid_form"
        data["form_errors"] = form.errors
    
    if "error" not in data:
        result_pk = form.cleaned_data['result']
        support_upload = models.SupportUpload.objects.filter(result=result_pk).order_by('-id').first()
        if not support_upload:
            data["created"] = True
            support_upload = models.SupportUpload(
                account = account,
                result_id = result_pk,
                user = request.user,
                local_status = "Preparing",
                contact_email = form.cleaned_data['contact_email'],
                description = form.cleaned_data['description']
            )
            support_upload.save()
        
        async_result = AsyncResult(support_upload.celery_task_id)
        if (not support_upload.celery_task_id 
        or async_result.status in celery.states.READY_STATES):
            if not support_upload.file:
                monitor = models.FileMonitor()
                monitor.save()
                support_upload.file = monitor
                support_upload.save()
            support_upload.file.status = "Queued",
            support_upload.file.tags = "support_upload,generate_csa"
            support_upload.file.save()

            generate = generate_csa.s(result_pk, support_upload.file.pk)
            errback = filemonitor_errback.s(support_upload.file.pk)
            gen_result = generate.apply_async(link_error=errback)
            support_upload.file.celery_task_id = gen_result.task_id

            auth_result = check_authentication.delay(support_upload.pk)
            upload_result = check_and_upload.apply_async((support_upload.pk, auth_result.task_id, gen_result.task_id), countdown=1)

            support_upload.celery_task_id = upload_result.task_id
            support_upload.save()

        resource = SupportUploadResource()
        uri = resource.get_resource_uri(support_upload)
        data["resource_uri"] = uri
    # make CSA
    # check auth
    # start upload task
    # respond with support upload object for JS polling
    return HttpResponse(json.dumps(data, indent=4, sort_keys=True), mimetype='application/json')

