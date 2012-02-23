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

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render_to_response
from django import forms


from iondb.rundb import models
from celery.task import task

try:
    import json
except ImportError:
    import simplejson as json


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


def write_file(file_data, destination):
    """Write Django uploaded file object to disk incrementally in order to
    avoid sucking up all of the system's RAM by reading the whole thing in to
    memory at once.
    """
    out = open(destination, 'wb+')
    for chunk in file_data.chunks():
        out.write(chunk)
    out.close()


def store_upload(pub, file_data, file_name, meta_data=None):
    """Create a unique folder for an uploaded file and begin editing it for
    publication.
    """
    upload = models.ContentUpload()
    upload.status = "Saving"
    upload.publisher = pub
    upload.meta = meta_data
    upload.save()
    pub_uploads = os.path.join("/results/uploads", pub.name)
    upload_dir = os.path.join(pub_uploads, str(upload.pk))
    os.makedirs(upload_dir)
    upload.file_path = os.path.join(upload_dir, file_name)
    write_file(file_data, upload.file_path)
    # TODO: Any path's defined here should really be persisted with the Content Upload
    meta_path = os.path.join(upload_dir, "meta.json")
    open(meta_path, 'w').write(meta_data)
    upload.status = "Queued for processing"
    upload.save()
    return upload


def run_script(working_dir, script_path, upload_id, upload_dir, upload_path, meta_path):
    """Run a Publisher's editing script with the uploaded file's information.
    working_dir = the Publisher's script directory.
    """
    script = os.path.basename(script_path)
    cmd = [script_path, upload_id, upload_dir, upload_path, meta_path]
    # Spawn the test subprocess and wait for it to complete.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir)
    result = proc.wait()
    stdout, stderr = proc.communicate()
    stdout_log = os.path.join(upload_dir, "%s_standard_output.log" % script)
    stderr_log = os.path.join(upload_dir, "%s_standard_error.log" % script)
    open(stdout_log, 'w').write(stdout)
    if stderr:
        open(stderr_log, 'w').write(stderr)
    return result, stdout, stderr


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
            result, stdout, stderr = run_script(pub_dir, script_path, str(upload.id),
                                                upload_dir, upload_path, meta_path)
            # The script may have updated the upload during execution, so we reload
            upload = models.ContentUpload.objects.get(pk=upload.pk)
            if result is 0:
                logger.info("Editing upload for %s finished %s" % (pub.name, script_path))
            else:
                logger.error("Editing for %s died during %s." % (pub.name, script_path))
                upload.status = "Error: publisher failed."
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



def edit_upload(pub, upload, meta=None):
    """Editing is the process which converts an uploaded file into one or more
    files of published content.
    """
    upload = store_upload(pub, upload, upload.name, meta)
    async_upload = run_pub_scripts.delay(pub, upload)
    return upload, async_upload


def upload_view(request):
    """Display a list of available publishers to upload files.
    """
    pubs = models.Publisher.objects.all()
    return render_to_response('rundb/ion_publisher_upload.html', {'pubs': pubs})


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
            # This is a gigantic placeholder.
            from django.http import HttpResponseRedirect
            return HttpResponseRedirect("/rundb/uploadstatus/frame/%d/" % upload.pk)
        else:
            logger.warning(form.errors)
    else:
        if frame:
            genome = request.GET.get('genome',False)
            uploader = os.path.join(pub.path, "upload.html")
            return render_to_response(uploader, {"genome": genome})
    return render_to_response("rundb/ion_publisher_frame.html", {"pub":pub})


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


def list_content(request):
    publishers = models.Publisher.objects.all()
    pubs = dict((p.name, list(p.contents.all())) for p in publishers)
    return render_to_response('rundb/ion_publisher_list_content.html',
                    {"pubs": pubs})
