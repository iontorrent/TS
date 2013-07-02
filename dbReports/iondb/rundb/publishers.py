# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Tasks
=====

The ``publishers`` module contains Django views and their helper functions
related to the processing if publisher uploads.

Not all functions contained in ``publishers`` are actual Django views, only
those that take ``request`` as their first argument and appear in a ``urls``
module are in fact Django views.
"""


import datetime
import subprocess
import logging
import traceback
import os
import os.path
import httplib
import mimetypes
import shutil

from tastypie.bundle import Bundle

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django import forms
from django.contrib.auth.decorators import login_required

from iondb.rundb import models
from celery.task import task

import json
from ajax import render_to_json


logger = logging.getLogger(__name__)


# ============================================================================
# Publisher Management
# ============================================================================


def search_for_publishers(config, pub_dir="/results/publishers/"):
    """
    Searches for new publishers, reads their publisher_meta.json, and makes any
    necessary updates to the publisher's record.
    """
    def create_new(name, version, path):
        pub = models.Publisher()
        pub.name = name
        pub.version = version
        pub.date = datetime.datetime.now()
        pub.path = path
        pub.save()

    def update_version(pub, version):
        pub.version = version
        pub.save()

    if os.path.exists(pub_dir):
        # only list files in the 'publishers' directory if they are actually folders
        folder_list = [i for i in os.listdir(pub_dir) if (
                        os.path.isdir(os.path.join(pub_dir, i)) and i != "scratch")]
        for pname in folder_list:
            full_path = os.path.join(pub_dir, pname)
            pub_meta_path = os.path.join(full_path, "publisher_meta.json")
            try:
                with open(pub_meta_path) as pub_meta_file:
                    publisher_meta = json.load(pub_meta_file)
                version = str(publisher_meta["version"])
            # Begin Righteous error reporting!
            except NameError:
                logger.error("Publisher %s is missing publisher_meta.json" % pname)
            except IOError as error:
                logger.error("Publisher %s failed to read publisher_meta.json with %s" % (pname, error))
            except (ValueError, KeyError) as error:
                logger.error("Publisher %s has an improperly formatted publisher_meta.json with %s" % (pname, error))
            else:
                try:
                    p = models.Publisher.objects.get(name=pname.strip())
                    if p.version != version:
                        update_version(p, version)
                        logger.info("Publisher %s updated to version %s" % (pname, version))
                except ObjectDoesNotExist:
                    create_new(pname, version, full_path)
                    logger.info("Publisher %s version %s added" % (pname, version))


def purge_publishers():
    """Removes records from publisher table which no longer have corresponding
    folder on the file system.  If the folder does not exist, we assume that
    the publisher has been deleted.  In any case, one cannot execute the
    publisher if the publisher's folder has been removed.
    """
    pubs = models.Publisher.objects.all()
    # for each record, test for corresponding folder
    for pub in pubs:
        # if folder does not exist
        if not os.path.isdir(pub.path):
            # remove this record
            pub.delete()
            logger.info("Deleting publisher %s which no longer exists at %s" % (pub.name, pub.path))


# ============================================================================
# Content Upload Publication
# ============================================================================


class ContentUploadFileForm(forms.Form):
    publisher = forms.ModelChoiceField(queryset=models.Publisher.objects.all())
    file  = forms.FileField()
    meta = forms.CharField(widget=forms.HiddenInput)


class PublisherContentUploadValidator(forms.Form):
    file  = forms.FileField()
    meta = forms.CharField(widget=forms.HiddenInput)


def write_file(file_data, destination):
    """Write Django uploaded file object to disk incrementally in order to
    avoid sucking up all of the system's RAM by reading the whole thing in to
    memory at once.
    """
    out = open(destination, 'wb+')
    for chunk in file_data.chunks():
        out.write(chunk)
    out.close()

@login_required
def write_plupload(request, pub_name):
    """file upload for plupload"""
    logger.info("Starting write plupload")

    pub = models.Publisher.objects.get(name=pub_name)
    logger.debug("%s %s" % (str(type(request.REQUEST['meta'])), request.REQUEST['meta']))

    logger.debug("Publisher Plupload started")
    if request.method == 'POST':
        name = request.REQUEST.get('name','')
        uploaded_file = request.FILES['file']
        if not name:
            name = uploaded_file.name
        logger.debug("plupload name = '%s'" % name)

        #check to see if a user has uploaded a file before, and if they have
        #not, make them a upload directory

        upload_dir = "/results/referenceLibrary/temp"

        if not os.path.exists(upload_dir):
            return render_to_json({"error":"upload path does not exist"})

        dest_path = os.path.join(upload_dir, name)

        logger.debug("plupload destination = '%s'" % dest_path)

        chunk = request.REQUEST.get('chunk','0')
        chunks = request.REQUEST.get('chunks','0')

        logger.debug("plupload chunk %s %s of %s" % (str(type(chunk)), str(chunk), str(chunks)))

        debug = [chunk, chunks]

        with open(dest_path,('wb' if chunk=='0' else 'ab')) as f:
            for content in uploaded_file.chunks():
                logger.debug("content chunk = '%d'" % len(content))
                f.write(content)

        if int(chunk) + 1 >= int(chunks):
            try:
                upload = move_upload(pub, dest_path, name, request.REQUEST['meta'])
                async_upload = run_pub_scripts.delay(pub, upload)
            except Exception as err:
                logger.exception("There was a problem during upload of a file for a publisher.")
            else:
                logger.info("Successfully pluploaded %s" % name)

        logger.debug("plupload done")
        return render_to_json({"chunk posted": debug})

    else:
        return render_to_json({"method":"only post here"})


def new_upload(pub, file_name, meta_data=None):
    upload = models.ContentUpload()
    upload.status = "Saving"
    upload.publisher = pub
    upload.meta = meta_data
    upload.save()
    pub_uploads = os.path.join("/results/uploads", pub.name)
    upload_dir = os.path.join(pub_uploads, str(upload.pk))
    os.makedirs(upload_dir)
    upload.file_path = os.path.join(upload_dir, file_name)
    # TODO: Any path's defined here should really be persisted with the Content Upload
    meta_path = os.path.join(upload_dir, "meta.json")
    open(meta_path, 'w').write(meta_data)
    upload.save()
    return upload


def move_upload(pub, file_path, file_name, meta_data=None):
    upload = new_upload(pub, file_name, meta_data)
    shutil.move(file_path, upload.file_path)
    upload.status = "Queued for processing"
    upload.save()
    return upload


def store_upload(pub, file_data, file_name, meta_data=None):
    """Create a unique folder for an uploaded file and begin editing it for
    publication.
    """
    upload = new_upload(pub, file_name, meta_data)
    write_file(file_data, upload.file_path)
    upload.status = "Queued for processing"
    upload.save()
    return upload


def run_script(working_dir, script_path, upload_id, upload_dir, upload_path, meta_path):
    """Run a Publisher's editing script with the uploaded file's information.
    working_dir = the Publisher's script directory.
    """
    script = os.path.basename(script_path)
    cmd = [script_path, upload_id, upload_dir, upload_path, meta_path]
    print(str(cmd))
    # Spawn the test subprocess and wait for it to complete.
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir)
        stdout, stderr = proc.communicate()
        result = proc.returncode
        stdout_log = os.path.join(upload_dir, "%s_standard_output.log" % script)
        stderr_log = os.path.join(upload_dir, "%s_standard_error.log" % script)
        open(stdout_log, 'w').write(stdout)
        if stderr:
            open(stderr_log, 'w').write(stderr)
    except Exception as err:
        print("Publisher error in upload %s: %s" % (upload_id, str(cmd)))
        raise
        return False
    return result == 0


@task
def run_pub_scripts(pub, upload):
    """Spawn subshells in which the Publisher's editing scripts are run, with
    the upload's folder and the script's output folder as command line args.
    """
    try:
        logger = run_pub_scripts.get_logger()
        #TODO: Handle unique file upload instance particulars
        logger.info("Editing upload for %s" % pub.name)
        previous_status = upload.status
        upload_path = upload.file_path
        upload_dir = os.path.dirname(upload_path)
        meta_path = os.path.join(upload_dir, "meta.json")
        pub_dir = pub.path
        pub_scripts = pub.get_editing_scripts()
        for script_path, stage_name in pub_scripts:
            # If at some point in the loop, one of the scripts changes the status,
            # then we cease updating it automatically.
            if upload.status == previous_status:
                previous_status = stage_name
                upload.status = stage_name
                upload.save()
            success = run_script(pub_dir, script_path, str(upload.id),
                                                upload_dir, upload_path, meta_path)
            # The script may have updated the upload during execution, so we reload
            upload = models.ContentUpload.objects.get(pk=upload.pk)
            if success:
                logger.info("Editing upload for %s finished %s" % (pub.name, script_path))
            else:
                logger.error("Editing for %s died during %s." % (pub.name, script_path))
                upload.status = "Error: %s" % stage_name
                upload.save()
            # If either the script itself or we set the status to anything starting
            # with "Error" then we abort further processing here.
            if upload.status.startswith("Error"):
                return
        # At this point every script has finished running and we have not returned
        # early due to an error, alright!
        upload.status = "Successfully Completed"
        upload.save()
    except Exception as error:
        tb = "\n".join("    "+s for s in traceback.format_exc().split("\n"))
        logger.error("Exception in %s upload %d during %s\n%s" %
            (pub.name, upload.id, stage_name, tb))
        upload.status = "Error: processing failed."
        upload.save()

@login_required
def edit_upload(pub, upload, meta=None):
    """Editing is the process which converts an uploaded file into one or more
    files of published content.
    """
    upload = store_upload(pub, upload, upload.name, meta)
    async_upload = run_pub_scripts.delay(pub, upload)
    return upload, async_upload

@login_required
def upload_view(request):
    """Display a list of available publishers to upload files.
    """
    pubs = models.Publisher.objects.all()
    return render_to_response('rundb/ion_publisher_upload.html', {'pubs': pubs})

@login_required
def publisher_upload(request, pub_name, frame=False):
    """Display the publishers upload.html template on a GET of the page.
    If the view is POSTed to, the pass the uploaded data to the publisher.
    """
    pub = models.Publisher.objects.get(name=pub_name)
    if request.method == 'POST':
        form = PublisherContentUploadValidator(request.POST, request.FILES)
        if form.is_valid():
            upload, async = edit_upload(pub, form.cleaned_data['file'],
                                 form.cleaned_data['meta'])

        else:
            logger.warning(form.errors)
    else:
        if frame:
            genome = request.GET.get('genome',False)
            uploader = os.path.join(pub.path, "upload.html")
            return render_to_response(uploader, {"genome": genome})
    return render_to_response("rundb/ion_publisher_frame.html", {"pub":pub})

@login_required
def publisher_api_upload(request, pub_name):
    """TastyPie does not support file uploads, so for now, this is handled
    outside of the normal API space.
    """
    if request.method == 'POST':
        pub = models.Publisher.objects.get(name=pub_name)
        form = PublisherContentUploadValidator(request.POST, request.FILES)
        if form.is_valid():
            upload, async = edit_upload(pub, form.cleaned_data['file'],
                                        form.cleaned_data['meta'])
            from iondb.rundb.api import ContentUploadResource
            resource = ContentUploadResource()
            bundle = Bundle(upload)
            serialized_upload = resource.serialize(None,
                                            resource.full_dehydrate(bundle),
                                           "application/json")
            return HttpResponse(serialized_upload,
                                mimetype="application/json")
        else:
            logger.warning(form.errors)
    else:
        return HttpResponseRedirect("/rundb/publish/%s/" % pub_name)

@login_required
def upload_status(request, contentupload_id, frame=False):
    """If we're in an iframe, we can skip basically everything, and tell the
    template to redirect the parent window to the normal page.
    """
    if frame:
        return render_to_response('rundb/ion_jailbreak.html',
                {"go": "/rundb/uploadstatus/%s/" % contentupload_id,
                 "contentupload_id": contentupload_id})
    upload = models.ContentUpload.objects.get(pk=contentupload_id)
    filename = os.path.basename(upload.file_path)
    logs = list(upload.logs.all())
    logs.sort(key=lambda x: x.timeStamp)
    return render_to_response('rundb/ion_publisher_upload_status.html',
                {"contentupload": upload, "logs": logs, "filename": filename})

@login_required
def list_content(request):
    publishers = models.Publisher.objects.all()
    pubs = dict((p.name, list(p.contents.all())) for p in publishers)
    return render_to_response('rundb/ion_publisher_list_content.html',
                    {"pubs": pubs})


def post_multipart(host, selector, fields, files):
    """Post fields and files to an http host as multipart/form-data.
    fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files
    Return the server's response page.
    """
    content_type, body = encode_multipart_formdata(fields, files)
    h = httplib.HTTP(host)
    h.putrequest('POST', selector)
    h.putheader('content-type', content_type)
    h.putheader('content-length', str(len(body)))
    h.endheaders()
    h.send(body)
    errcode, errmsg, headers = h.getreply()
    return h.file.read()


def encode_multipart_formdata(fields, files):
    """fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files
    Return (content_type, body) ready for httplib.HTTP instance
    """
    BOUNDARY = 'GlobalNumberOfPiratesDecreasing-GlobalTemperatureIncreasing'
    CRLF = '\r\n'
    request = []
    for (key, value) in fields:
        request.extend([
            '--' + BOUNDARY,
            'Content-Disposition: form-data; name="%s"' % key,
            '',
            value
        ])
    for (key, filename, value) in files:
        request.extend([
            '--' + BOUNDARY,
            'Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename),
            'Content-Type: %s' % get_content_type(filename),
            '',
            value
        ])
    request.append('--' + BOUNDARY + '--')
    request.append('')
    body = CRLF.join(request)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body


def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'
