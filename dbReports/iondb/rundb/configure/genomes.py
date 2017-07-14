# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import csv
import datetime
import os
import shutil
import socket
import xmlrpclib
import glob
import fileinput
import logging
import json
import base64
import httplib2
import tempfile
import zipfile
import mimetypes
import re
import subprocess
import urllib2
from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponse, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template import RequestContext
from django.conf import settings
from django.core import urlresolvers
from django.core.files import File

from iondb.rundb.ajax import render_to_json
from iondb.rundb.forms import EditReferenceGenome
from iondb.rundb.models import ReferenceGenome, ContentUpload, FileMonitor, Publisher
from iondb.rundb import tasks, publishers

from iondb.rundb.configure.util import plupload_file_upload

logger = logging.getLogger(__name__)

JOBSERVER_HOST = "127.0.0.1"


@login_required
def file_upload(request):
    return plupload_file_upload(request, settings.TEMP_PATH)


@login_required
def delete_genome(request, pk):
    """delete a reference genome
    the filesystem file deletions should be done with a method on the model"""

    if request.method == "POST":
        ref_genome = get_object_or_404(ReferenceGenome, pk=pk)

        #delete dir by default
        try_delete = ref_genome.delete()

        if not try_delete:
            #the file could not be deleted, present the user with an error message.
            return render_to_json({"status": " <strong>Error</strong> <p>Genome could not be deleted.</p> \
                                          <p>Check the file permissions for the genome on the file system at: </p> \
                                          <p><strong>" + str(ref_genome.reference_path) + "</p></strong> "})

        return render_to_json({"status": "Genome was deleted successfully"})

    if request.method == "GET":
        return render_to_json({"status": "This must be accessed via post"})


def _change_genome_name(rg, new_name, old_full_name, new_full_name):
    """
    to change a genome name, we have to
    1) move the dir from the old name to the new one
    2) change the data in <genome_name>.info.txt
    3) change all of the files names to have a prefix of the new genome name
    4) rewrite the reference_list.txt files, which is being done from the calling function
    """

    if (new_name != rg.short_name):
        #we also really need to check to see if the file exsits.
        old_files = glob.glob(rg.reference_path + "/" + rg.short_name + ".*")

        def rreplace(s, old, new, occurrence):
            """replace from starting at the right"""
            li = s.rsplit(old, occurrence)
            return new.join(li)

        for old_file in old_files:
            os.rename(old_file, rreplace(old_file, rg.short_name, new_name, 1))

        shutil.move(rg.reference_path, settings.TMAP_DIR + new_name)

    info = os.path.join(settings.TMAP_DIR, new_name, new_name + ".info.txt")

    #this will rewrite the genome.info.text file
    for line in fileinput.input(info, inplace=1):
        if line.split('\t')[0] == "genome_name":
            print line.replace(old_full_name, new_full_name),
        else:
            print line,


def _write_genome_info(info_path, _dict):
    """write genome info to file from dict
    """
    try:
        genome_info = open(info_path, 'w')
        for key, value in _dict.items():
            genome_info.write(str(key))
            genome_info.write("\t")
            genome_info.write(str(value))
            genome_info.write("\n")

    except IOError:
        return False

    genome_info.close()

    return True


def _read_genome_info(info_path):
    """each genome has a genome.info.txt file which contains meta info for that genome
    here we will find and return that as a string
    if False is returned the genome can be considered broken
    """
    #build a dict with the values from the info.txt
    genome_dict = {"genome_name": None, "genome_version": None, "index_version": None}
    try:
        for line in csv.reader(open(info_path), dialect='excel-tab'):
            if len(line) == 2:
                genome_dict[line[0]] = line[1]
    except IOError as err:
        logger.error("Could not read genome info file '{0}': {1}".format(info_path, err))
        return None

    return genome_dict


def _genome_get_fasta(pk):
    """each genome should have a fasta file
        check if that exists, and if it does return a link
        we also have to provide someway to download the fasta files with apache
    """

    try:
        rg = ReferenceGenome.objects.get(pk=pk)
    except Exception as err:
        logger.exception("Error reading finding genome_fasta path")
        return False

    genome_fasta = os.path.join(rg.reference_path, rg.short_name + ".fasta")
    size = None
    if os.path.exists(genome_fasta):
        size = os.path.getsize(genome_fasta)
    else:
        genome_fasta = False

    return genome_fasta, size


def _verbose_error_trim(verbose_error):
    """try to make verbose error messages a bit easier for humans to read"""
    try:
        verbose_error = json.loads(verbose_error)
        verbose_error = verbose_error[1:-1]
    except:
        return False

    if "validate_reference" in verbose_error[0]:
        pretty = ["FASTA file failed validation. Please review the error below and modify the FASTA file to correct the problem."]
        try:
            lines = verbose_error[0].split('\n\n')
            pretty.append(lines[-2].split(": ")[1])
            pretty.append(lines[1])
            return pretty
        except:
            return verbose_error

    return verbose_error


@login_required
def edit_genome(request, pk_or_name):
    """Make changes to an existing genome database reference,
    or create a new one if ``pk`` is zero."""
    try:
        rg = ReferenceGenome.objects.get(pk=pk_or_name)
    except (ValueError, ReferenceGenome.DoesNotExist):
        rg = get_object_or_404(ReferenceGenome, short_name=pk_or_name)

    uploads = ContentUpload.objects.filter(publisher__name="BED")
    relevant = [u for u in uploads if u.meta.get("reference", "") == rg.short_name]
    #TODO give an indication if it is a hotspot BED file
    bedFiles, processingBedFiles = [], []
    for upload in relevant:
        info = {"path": os.path.basename(upload.file_path), "pk": upload.pk}
        if upload.status == "Successfully Completed":
            bedFiles.append(info)
        else:
            info["status"] = upload.status
            processingBedFiles.append(info)
    bedFiles.sort(key=lambda x: x['path'].lower())
    processingBedFiles.sort(key=lambda x: x['path'].lower())

    #Annotation Details Page
    refAnnotFiles, processingRefAnnotFiles = [], []
    try:
        uploadsAnnot = ContentUpload.objects.filter(publisher__name="refAnnot")
    except:
        uploadsAnnot = None
    if uploadsAnnot:
        relevantAnnot = [u for u in uploadsAnnot if u.meta.get("reference", "") == rg.short_name]
        for uploadAnnot in relevantAnnot:
            annotInfo = {"path": os.path.basename(uploadAnnot.file_path), "pk": uploadAnnot.pk}
            if uploadAnnot.status == "Successfully Completed":
                refAnnotFiles.append(annotInfo)
            else:
                annotInfo["status"] = uploadAnnot.status
                processingRefAnnotFiles.append(annotInfo)
        refAnnotFiles.sort(key=lambda x: x['path'].lower())
        processingRefAnnotFiles.sort(key=lambda x: x['path'].lower())

    if request.method == "POST":
        rfd = EditReferenceGenome(request.POST)
        if rfd.is_valid():
            rg.notes = rfd.cleaned_data['notes']
            rg.enabled = rfd.cleaned_data['enabled']
            rg.date = datetime.datetime.now()

            if (rg.short_name != rfd.cleaned_data['name'] or rg.name != rfd.cleaned_data['NCBI_name']):
                _change_genome_name(rg, rfd.cleaned_data['name'], rg.name, rfd.cleaned_data['NCBI_name'])

            #make sure to only set the new name after the _change_genome_name call - it needs the old full name
            rg.name = rfd.cleaned_data['NCBI_name']
            rg.short_name = rfd.cleaned_data['name']

            #Update the reference path
            if rg.enabled:
                rg.enable_genome()
            else:
                rg.disable_genome()
            rg.save()

            url = urlresolvers.reverse("configure_references")
            return HttpResponsePermanentRedirect(url)
        else:
            genome_dict = _read_genome_info(rg.info_text())
            verbose_error = _verbose_error_trim(rg.verbose_error)
            genome_fasta, genome_size = _genome_get_fasta(rg.pk)

            ctxd = {"temp": rfd, "name": rg.short_name, "reference": rg, "key": rg.pk, "enabled": rg.enabled,
                    "genome_dict": genome_dict, "status": rg.status, "verbose_error": verbose_error,
                    "genome_fasta": genome_fasta, "genome_size": genome_size,
                    "bedFiles": bedFiles, "processingBedFiles": processingBedFiles,
                    "index_version": rg.index_version
                    }
            ctx = RequestContext(request, ctxd)
            return render_to_response("rundb/configure/edit_reference.html",
                                      context_instance=ctx)
    elif request.method == "GET":
        temp = EditReferenceGenome()
        temp.fields['NCBI_name'].initial = rg.name
        temp.fields['name'].initial = rg.short_name
        temp.fields['notes'].initial = rg.notes
        temp.fields['enabled'].initial = rg.enabled
        temp.fields['genome_key'].initial = rg.pk
        temp.fields['index_version'].initial = rg.index_version

        genome_dict = _read_genome_info(rg.info_text()) or {}
        genome_fasta, genome_size = _genome_get_fasta(rg.pk)

        verbose_error = _verbose_error_trim(rg.verbose_error)
        fastaOrig = rg.fastaOrig()

        stale_index = rg.index_version != settings.TMAP_VERSION and rg.status != "Rebuilding index"

        ctxd = {"temp": temp, "name": rg.short_name, "reference": rg, "key": rg.pk, "enabled": rg.enabled,
                "genome_dict": genome_dict, "status": rg.status, "verbose_error": verbose_error,
                "genome_fasta": genome_fasta, "genome_size": genome_size,
                "index_version": rg.index_version, "fastaOrig": fastaOrig,
                "bedFiles": bedFiles, "processingBedFiles": processingBedFiles,
                "stale_index": stale_index,
                "refAnnotFiles": refAnnotFiles, "processingRefAnnotFiles": processingRefAnnotFiles,
                }
        ctx = RequestContext(request, ctxd)
        return render_to_response("rundb/configure/edit_reference.html",
                                  context_instance=ctx)


@login_required
def genome_status(request, reference_id):
    """Provide a way for the index creator to let us know when the index has been created"""

    if request.method == "POST":
        rg = get_object_or_404(ReferenceGenome, pk=reference_id)
        status = request.POST.get('status', False)
        enabled = request.POST.get('enabled', False)
        verbose_error = request.POST.get('verbose_error', "")
        index_version = request.POST.get('index_version', "")

        if not status:
            return render_to_json({"status": "error genome status not given"})

        rg.status = status
        rg.enabled = enabled
        rg.verbose_error = verbose_error
        rg.index_version = index_version
        rg.reference_path = os.path.join(settings.TMAP_DIR, rg.short_name)

        rg.save()
        return render_to_json({"status": "genome status updated", "enabled": enabled})
    if request.method == "GET":
        rg = get_object_or_404(ReferenceGenome, pk=reference_id)
        return render_to_json({"status": rg.status})


def search_for_genomes():
    """
    Searches for new genomes.  This will sync the file system and the genomes know by the database
    """
    def set_common(dest, genome_dict, ref_dir, lib):
        try:
            dest.name = genome_dict["genome_name"]
            dest.version = genome_dict["genome_version"]
            dest.index_version = genome_dict["index_version"]
            dest.reference_path = os.path.join(ref_dir, dest.index_version, dest.short_name)
        except:
            dest.name = lib
            dest.status = "missing info.txt"
        return dest

    ref_dir = '/results/referenceLibrary'

    lib_versions = []

    for folder in os.listdir(ref_dir):
        if os.path.isdir(os.path.join(ref_dir, folder)) and folder.lower().startswith("tmap"):
            lib_versions.append(folder)
    logger.debug("Reference genome scanner found %s" % ",".join(lib_versions))
    for lib_version in lib_versions:
        if os.path.exists(os.path.join(ref_dir, lib_version)):
            libs = os.listdir(os.path.join(ref_dir, lib_version))
            for lib in libs:
                genome_info_text = os.path.join(ref_dir, lib_version, lib, lib + ".info.txt")
                genome_dict = _read_genome_info(genome_info_text)
                #TODO: we have to take into account the genomes that are queue for creation of in creation

                if genome_dict:
                    #here we trust that the path the genome is in, is also the short name
                    existing_reference = ReferenceGenome.objects.filter(
                        short_name=lib).order_by("-index_version")[:1]
                    if existing_reference:
                        rg = existing_reference[0]
                        if rg.index_version != genome_dict["index_version"]:
                            logger.debug("Updating genome status to 'found' for %s id=%d index=%s" % (
                            str(rg), rg.id, rg.index_version))
                            rg.status = "complete"
                            rg = set_common(rg, genome_dict, ref_dir, lib)
                            rg.save()
                    else:
                        logger.info("Found new genome %s index=%s" % (
                            lib, genome_dict["genome_version"]))
                        #the reference was not found, add it to the db
                        rg = ReferenceGenome()
                        rg.short_name = lib
                        rg.date = datetime.datetime.now()
                        rg.status = "complete"
                        rg.enabled = True

                        rg.index_version = ""
                        rg.version = ""
                        rg.name = ""

                        rg = set_common(rg, genome_dict, ref_dir, lib)

                        rg.save()
                        logger.info("Created new reference genome %s id=%d" % (
                            str(rg), rg.id))


@login_required
def add_custom_genome(request):
    ''' Import custom genome via file upload or URL '''
    if request.method == "POST":
        url = request.POST.get("reference_url", None)
        target_file = request.POST.get('target_file', False)
        reference_args = {
            "short_name": request.POST.get("short_name"),
            "name": request.POST.get("name"),
            "version": request.POST.get("version", ""),
            "notes": request.POST.get("notes", ""),
            "index_version": ""
        }

        if target_file:
            # import via file upload
            reference_path = os.path.join(settings.TEMP_PATH, target_file)
            reference_args["source"] = reference_path

            # check expected file size
            failed = False
            reported_file_size = request.POST.get('reported_file_size', False)
            try:
                uploaded_file_size = str(os.path.getsize(reference_path))
                if reported_file_size and (reported_file_size != uploaded_file_size):
                    failed = "Upload error: uploaded file size is incorrect"
            except OSError:
                failed = "Upload error: temporary file not found"

            try:
                new_reference_genome(reference_args, None, target_file)
            except Exception as e:
                failed = str(e)

            if failed:
                try:
                    os.remove(reference_path)
                except OSError:
                    failed += " The FASTA file could not be deleted."

                logger.error("Failed uploading genome file: " + failed)
                return render_to_json({"status": failed, "error": True})
            else:
                return render_to_json({"error": False})
        else:
            # import via URL
            reference_args["source"] = url
            try:
                new_reference_genome(reference_args, url, None)
            except Exception as e:
                return render_to_json({"status": str(e), "error": True})
            else:
                return HttpResponseRedirect(urlresolvers.reverse("configure_references"))

    elif request.method == "GET":
        return render_to_response("rundb/configure/modal_references_new_genome.html", context_instance=RequestContext(request, {}))


@login_required
def start_index_rebuild(request, reference_id):
    def rebuild_index(reference):
        """Add a job to rebuild the reference index for reference to the SGE queue
        """
        logger.info("Queuing TMAP reference index rebuild of %s" % reference.short_name)
        reference.status = "indexing"
        result = tasks.build_tmap_index.delay(reference.id)
        reference.celery_task_id = result.task_id
        reference.save()
    data = {"references": []}
    if reference_id == "all":
        references = ReferenceGenome.objects.exclude(index_version=settings.TMAP_VERSION)
        logger.info("Rebuilding TMAP reference indices for %s" %
                    ", ".join(r.short_name for r in references))
        for reference in references:
            rebuild_index(reference)
            data["references"].append({"id": reference.pk,
                                       "short_name": reference.short_name})
    else:
        reference = ReferenceGenome.objects.get(pk=reference_id)
        rebuild_index(reference)
        data["references"].append({"id": reference.pk,
                                   "short_name": reference.short_name})
    return HttpResponse(json.dumps(data), mimetype="application/json")


def get_references():
    h = httplib2.Http()
    response, content = h.request(settings.REFERENCE_LIST_URL)
    if response['status'] == '200':
        references = json.loads(content)
        id_hashes = [r['meta']['identity_hash'] for r in references if r['meta']['identity_hash']]
        installed = dict((r.identity_hash, r) for r in ReferenceGenome.objects.filter(identity_hash__in=id_hashes))
        for ref in references:
            ref["meta_encoded"] = base64.b64encode(json.dumps(ref["meta"]))
            ref["notes"] = ref["meta"].get("notes", '')
            ref["installed"] = installed.get(ref['meta']['identity_hash'], None)
            ref["annotation_encoded"] = base64.b64encode(json.dumps(ref.get("annotation", "")))
            ref["reference_mask_encoded"] = ref.get("reference_mask", None)
            ref["bedfiles"] = ref.get("bedfiles", [])
        return references
    else:
        return None


def download_genome_annotation(annotation_lists, reference_args):
    for item in annotation_lists:
        remoteAnnotUrl = item.get("url", None)
        remoteAnnotUpdateVersion = item.get("updateVersion", None)

        if remoteAnnotUrl:
            tasks.new_annotation_download.delay(remoteAnnotUrl, remoteAnnotUpdateVersion, **reference_args)


@login_required
def download_genome(request):
    # called by "Import Preloaded Ion References"
    if request.method == "POST":
        reference_meta = request.POST.get("reference_meta", None)
        ref_annot_update = request.POST.get("ref_annot_update", None)
        reference_args = json.loads(base64.b64decode(reference_meta))
        annotation_meta = request.POST.get("missingAnnotation_meta", None)
        reference_mask_info = request.POST.get("reference_mask", None)

        if annotation_meta:
            annotation_data = json.loads(base64.b64decode(annotation_meta))
            annotation_lists = annotation_data

        # Download and register only the Ref Annotation file if Reference Genome is already Imported
        # If not, download refAnnot file and Reference Genome asynchronously
        if annotation_data and ref_annot_update:
            logger.debug("Downloading Annotation File {0} with meta {1}".format(annotation_data, reference_meta))
            if annotation_lists:
                download_genome_annotation(annotation_lists, reference_args)
        else:
            url = request.POST.get("reference_url", None)
            logger.debug("Downloading {0} with meta {1}".format(url, reference_meta))
            if url is not None:
                if annotation_lists:
                    download_genome_annotation(annotation_lists, reference_args)

                try:
                    new_reference_genome(reference_args, url, reference_mask_filename=reference_mask_info)
                except Exception as e:
                    return render_to_json({"status": str(e), "error": True})

        return HttpResponseRedirect(urlresolvers.reverse("references_genome_download"))

    elif request.method == "GET":
        references = get_references() or []
        downloads = FileMonitor.objects.filter(tags__contains="reference").order_by('-created')
        downloads_annot = FileMonitor.objects.filter(tags__contains="reference_annotation").order_by('-created')

        (references, downloads_annot) = get_annotation(references, downloads_annot)

        ctx = {
            'downloads': downloads,
            'downloads_annot': downloads_annot,
            'references': references
        }
        return render_to_response("rundb/configure/reference_download.html", ctx, context_instance=RequestContext(request))


def validate_annotation_url(remoteAnnotUrl):
    isValid = True
    ctx = {}
    try:
        req = urllib2.urlopen(remoteAnnotUrl)
        url = req.geturl()
    except urllib2.HTTPError, err:
        isValid = False
        if err.code == 404:
            logger.debug("HTTP Error: {0}. Please contact TS administrator  {1}".format(err, remoteAnnotUrl))
            ctx['msg'] = "HTTP Error: {0}. Please contact TS administrator.".format(err.msg)
        else:
            logger.debug("Error in validate_annotation_url({0})".format(err))
            ctx['msg'] = err
    except urllib2.URLError, err:
        logger.debug("Connection Error in validate_annotation_url: ({0})".format(err))
        isValid = False
        ctx['msg'] = "Connection Error. Please contact TS administrator.".format(err.reason)

    ctx['isValid'] = isValid
    return ctx


def get_annotation(references, downloads_annot):
    for ref in references:
        annotation_meta = ref.get('annotation_encoded', None)
        annotation_data = json.loads(base64.b64decode(annotation_meta))
        missingAnnotation = []

        if not annotation_data:
            ref["refAnnotNotAvailable"] = 'N/A'
        else:
            for item in annotation_data:
                remoteAnnotUrl = item.get("url", None)
                remoteAnnotUpdateVersion = item.get("updateVersion", None)
                newAnnotFlag = True
                if remoteAnnotUrl:
                    for download in downloads_annot:
                        isValid = True
                        isValidNetwork = True
                        if (download.url == remoteAnnotUrl):
                            ctx = validate_annotation_url(remoteAnnotUrl)
                            isValidNetwork = ctx['isValid']
                        try:
                            fm_pk = FileMonitor.objects.get(pk=download.id)
                        except Exception as Err:
                            logger.debug("get_annotation() Error: {0} {1}".format(remoteAnnotUrl, str(download.status)))
                            logger.debug(Err)
                        if isValidNetwork:
                            tagsInfo = download.tags
                            try:
                                updateVersion = tagsInfo.split("reference_annotation_")[1]
                            except:
                                logger.debug("get_annotation() Error: {0} {1}".format(remoteAnnotUrl, str(download.status)))
                                updateVersion = None
                            download.updateVersion = updateVersion
                            filemonitor_status = fm_pk.status
                            filemonitor_status = filemonitor_status.lower()
                            if (filemonitor_status == 'downloading'):
                                newAnnotFlag = False
                            # Check if there is any newer version of annotation file has been posted
                            if ((fm_pk.url == remoteAnnotUrl)  and (float(updateVersion) >= float(remoteAnnotUpdateVersion))):
                                newAnnotFlag = False
                                if fm_pk.status == "Complete":
                                    ref['isAnnotCompleted'] = "complete"
                                else:
                                    isValid = True
                                    if 'Error' in str(download.status) or 'Error' in str(fm_pk.status):
                                        logger.debug("Error: {0} {1}".format(remoteAnnotUrl, str(download.status)))
                                        isValid = False
                                        ctx['msg'] = fm_pk.status
                                    if (filemonitor_status not in ["starting", "downloading", "download failed"]):
                                        logger.debug("get_annotation() Error: Status is not in the expected state: {0}".format(fm_pk.status))
                                        if ((not download.status) and (not fm_pk.status)):
                                            # Query FileMonitor and Get the status if there was any lag in the network/celery task
                                            fm_pk = FileMonitor.objects.get(pk=download.id)
                                            status = str(fm_pk.status)
                                            if (not fm_pk.status):
                                                isValid = False
                                                download.status = "Connection error. Please contact Torrent Suite Administrator"
                                                ctx['msg'] = fm_pk.status
                                            else:
                                                status = status.replace("Complete", "complete")
                                            ref['isAnnotCompleted'] = status
                                        else:
                                            ref['isAnnotCompleted'] = str(fm_pk.status)
                                    else:
                                        ref['isAnnotCompleted'] = str(fm_pk.status)
                        if not isValidNetwork or not isValid:
                            newAnnotFlag = False
                            ref['noLink'] = "True"
                            ref['noLinkMsg'] = ctx['msg']
                            ref['errURL'] = download.url
                            try:
                                if fm_pk:
                                    fm_pk.delete()
                            except Exception as err:
                                logger.debug("get_annotation() Error: {0} {1}".format(remoteAnnotUrl, str(download.status)))
                            ref['isAnnotCompleted'] = str(fm_pk.status)

                    if newAnnotFlag:
                        missingAnnotation.append(item)
                        ref['isNewAnnotPosted'] = remoteAnnotUrl
                        ref['isAnnotCompleted'] = None

        ref["missingAnnotation_meta"] = base64.b64encode(json.dumps(missingAnnotation))
        ref["missingAnnotation_data"] = json.loads(base64.b64decode(ref["missingAnnotation_meta"]))

    return (references, downloads_annot)


def new_reference_genome(reference_args, url=None, reference_file=None, callback_task=None, reference_mask_filename=None):
    # check if the genome already exists
    if ReferenceGenome.objects.filter(short_name=reference_args['short_name'], index_version=settings.TMAP_VERSION):
        raise Exception("Failed - Genome %s already exists" % reference_args['short_name'])

    reference = ReferenceGenome(**reference_args)
    reference.enabled = False
    reference.status = "queued"
    reference.save()

    if url:
        async_result = start_reference_download(url, reference, callback_task, reference_mask_filename=reference_mask_filename)
    elif reference_file:
        async_result = tasks.install_reference.apply_async(((reference_file, None), reference.id), link=callback_task)
    else:
        raise Exception('Failed creating new genome reference: No source file')

    reference.celery_task_id = async_result.task_id
    reference.save()
    return reference


def start_reference_download(url, reference, callback=None, reference_mask_filename=None):
    monitor = FileMonitor(url=url, tags="reference")
    monitor.save()
    reference.status = "downloading"
    reference.file_monitor = monitor
    reference.save()
    try:
        download_args = (url, monitor.id, settings.TEMP_PATH)
        install_callback = tasks.install_reference.subtask((reference.id, reference_mask_filename))
        if callback:
            install_callback.link(callback)
        async_result = tasks.download_something.apply_async(download_args, link=install_callback)
        return async_result
    except Exception as err:
        monitor.status = "System Error: " + str(err)
        monitor.save()

