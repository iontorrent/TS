# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
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

from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponse, HttpResponsePermanentRedirect, HttpResponseRedirect
from django.template import RequestContext
from django.conf import settings
from django.core import urlresolvers

from iondb.rundb.ajax import render_to_json
from iondb.anaserve import client
from iondb.rundb.forms import EditReferenceGenome
from iondb.rundb.models import ReferenceGenome, ContentUpload, DownloadMonitor
from iondb.rundb import tasks
from iondb.rundb.tasks import build_tmap_index
from iondb.rundb.configure.util import plupload_file_upload

logger = logging.getLogger(__name__)

JOBSERVER_HOST = "127.0.0.1"
REFERENCE_LIBRARY_TEMP_DIR = "/results/referenceLibrary/temp/"


def file_upload(request):
    return plupload_file_upload(request, REFERENCE_LIBRARY_TEMP_DIR)


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

    try:
        genome_info = open(info_path).readlines()

        #build a dict with the values from the info.txt
        genome_dict = {"genome_name": False, "genome_version": False, "index_version": False}
        for line in genome_info:
            line = line.strip().split("\t")
            try:
                genome_dict[line[0]] = line[1]
            except IndexError:
                genome_dict = False
                return genome_dict
    except IOError:
        genome_dict = False

    return genome_dict


def _genome_get_fasta(pk):
    """each genome should have a fasta file
        check if that exists, and if it does return a link
        we also have to provide someway to download the fasta files with apache
    """

    try:
        rg = ReferenceGenome.objects.get(pk=pk)
    except:
        return False

    genome_fasta = os.path.join(rg.reference_path, rg.short_name + ".fasta")

    size = False

    try:
        os.path.exists(genome_fasta)
        size = os.path.getsize(genome_fasta)
    except IOError:
        genome_fasta = False
    except os.error:
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
                enabled_path = os.path.join(settings.TMAP_DIR, rg.short_name)
                rg.enable_genome()
            else:
                base_path = os.path.split(os.path.split(rg.reference_path)[0])[0]
                enabled_path = os.path.join(base_path, "disabled", rg.index_version, rg.short_name)
                rg.disable_genome()

            rg.reference_path = enabled_path

            rg.save()

            url = urlresolvers.reverse("configure_references")
            return HttpResponsePermanentRedirect(url)
        else:
            genome_dict = _read_genome_info(rg.info_text())
            verbose_error = _verbose_error_trim(rg.verbose_error)
            genome_fasta, genome_size = _genome_get_fasta(rg.pk)

            ctxd = {"temp": rfd, "name": rg.short_name, "key": rg.pk, "enabled": rg.enabled,
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

        genome_dict = _read_genome_info(rg.info_text())
        genome_fasta, genome_size = _genome_get_fasta(rg.pk)

        verbose_error = _verbose_error_trim(rg.verbose_error)
        fastaOrig = rg.fastaOrig()

        stale_index = rg.index_version != settings.TMAP_VERSION and rg.status != "Rebuilding index"

        ctxd = {"temp": temp, "name": rg.short_name, "key": rg.pk, "enabled": rg.enabled,
                "genome_dict": genome_dict, "status": rg.status, "verbose_error": verbose_error,
                "genome_fasta": genome_fasta, "genome_size": genome_size,
                "index_version": rg.index_version, "fastaOrig": fastaOrig,
                "bedFiles": bedFiles, "processingBedFiles": processingBedFiles,
                "stale_index": stale_index
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
                logger.debug("Parsing reference genome info %s" % genome_info_text)
                genome_dict = _read_genome_info(genome_info_text)
                #TODO: we have to take into account the genomes that are queue for creation of in creation

                if genome_dict:
                    #here we trust that the path the genome is in, is also the short name
                    existing_reference = ReferenceGenome.objects.filter(
                        short_name=lib).order_by("-index_version")[:1]
                    if existing_reference:
                        rg = existing_reference[0]
                        logger.debug("Found existing genome %s id=%d index=%s" % (
                            str(rg), rg.id, rg.index_version))
                        if rg.index_version != genome_dict["index_version"]:
                            logger.debug("Updating genome status to 'found'")
                            rg.status = "found"
                            rg = set_common(rg, genome_dict, ref_dir, lib)
                            rg.save()
                    else:
                        logger.debug("Found new genome %s index=%s" % (
                            lib, genome_dict["genome_version"]))
                        #the reference was not found, add it to the db
                        rg = ReferenceGenome()
                        rg.short_name = lib
                        rg.date = datetime.datetime.now()
                        rg.status = "found"
                        rg.enabled = True

                        rg.index_version = ""
                        rg.version = ""
                        rg.name = ""

                        rg = set_common(rg, genome_dict, ref_dir, lib)

                        rg.save()
                        logger.info("Created new reference genome %s id=%d" % (
                            str(rg), rg.id))


@login_required
def new_genome(request):
    """This is the page to create a new genome. The XML-RPC server is ionJobServer.
    """

    if request.method == "POST":
        # parse the data sent in

        #required
        name = request.POST.get('name', False)
        short_name = request.POST.get('short_name', False)
        fasta = request.POST.get('target_file', False)
        version = request.POST.get('version', False)
        notes = request.POST.get('notes', "")

        #optional
        read_exclude_length = request.POST.get('read_exclude_length', False)

        #URL download
        url = request.POST.get('url', False)

        error_status = ""
        reference_path = REFERENCE_LIBRARY_TEMP_DIR + fasta

        why_delete = ""

        #if any of those were false send back a failed message
        if not all((name, short_name, fasta, version)):
            return render_to_json({"status": "Form validation failed", "error": True})

        if not set(short_name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"):
            return render_to_json({"status": "The short name has invalid characters. The valid values are letters, numbers, and underscores.", "error": True})

        #TODO: check to make sure the zip file only has one fasta or fa
        if not url:
            #check to ensure the size on the OS the same as the reported.
            reported_file_size = request.POST.get('reported_file_size', False)

            try:
                uploaded_file_size = str(os.path.getsize(reference_path))
            except OSError:
                return render_to_json({"status": "The FASTA temporary files was not found", "error": True})

            if reported_file_size != uploaded_file_size:
                why_delete = "The file you uploaded differs from the expected size. This is due to an error uploading."

            if not (fasta.lower().endswith(".fasta") or fasta.lower().endswith(".zip")):
                why_delete = "The file you uploaded does not have a .fasta or .zip extension.  It must be a plain text fasta file or a Zip compressed fasta."

            if why_delete:
                try:
                    os.remove(reference_path)
                except OSError:
                    why_delete += " The FASTA file could not be deleted."
                return render_to_json({"status": why_delete, "error": True})

        #Make an genome ref object
        if ReferenceGenome.objects.filter(short_name=short_name, index_version=settings.TMAP_VERSION):
            #check to see if the genome already exists in the database with the same version
            return render_to_json({"status": "Failed - Genome with this short name and index version already exist.", "error": True})
        ref_genome = ReferenceGenome()
        ref_genome.name = name
        ref_genome.short_name = short_name
        ref_genome.version = version
        ref_genome.date = datetime.datetime.now()
        ref_genome.notes = notes
        ref_genome.status = "queued"
        ref_genome.enabled = False
        ref_genome.index_version = settings.TMAP_VERSION

        #before the object is saved we should ping the xml-rpc server to see if it is alive.
        try:
            conn = client.connect(JOBSERVER_HOST, settings.JOBSERVER_PORT)
            #just check uptime to make sure the call does not fail
            conn.uptime()
            logger.debug('Connected to ionJobserver process.')
        except (socket.error, xmlrpclib.Fault):
            return render_to_json({"status": "Unable to connect to ionJobserver process.  You may need to restart ionJobserver", "error": True})

        #if the above didn't fail then we can save the object
        #this object must be saved before the tmap call is made
        ref_genome.save()
        logger.debug('Saved ReferenceGenome %s' % ref_genome.__dict__)

        #kick off the anaserve tmap xmlrpc call
        import traceback
        try:
            conn = client.connect(JOBSERVER_HOST, settings.JOBSERVER_PORT)
            tmap_bool, tmap_status = conn.tmap(str(ref_genome.id), fasta, short_name, name, version,
                                               read_exclude_length, settings.TMAP_VERSION)
            logger.debug('ionJobserver process reported %s %s' % (tmap_bool, tmap_status))
        except (socket.error, xmlrpclib.Fault):
            #delete the genome object, because it was not sucessful
            ref_genome.delete()
            return render_to_json({"status": "Error with index creation", "error": traceback.format_exc()})

        if not tmap_bool:
            ref_genome.delete()
            return render_to_json({"status": tmap_status, "error": True})

        return render_to_json({"status": "The genome index is being created.  This might take a while, check the status on the references tab. \
                                You are being redirected there now.", "error": False})

    elif request.method == "GET":
        ctx = RequestContext(request, {})
        return render_to_response("rundb/configure/modal_references_new_genome.html", context_instance=ctx)


@login_required
def start_index_rebuild(request, reference_id):
    def rebuild_index(reference):
        """Add a job to rebuild the reference index for reference to the SGE queue
        """
        logger.info("Queuing TMAP reference index rebuild of %s" % reference.short_name)
        reference.status = "Rebuilding index"
        reference.save()
        build_tmap_index.delay(reference)
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
    response, content = h.request("http://ionupdates.com/reference_downloads/references_list.json")
    if response['status'] == '200':
        references = json.loads(content)
        for ref in references:
            ref["meta"] = base64.b64encode(json.dumps(ref["meta"]))
        return references
    else:
        return None

@login_required
def download_genome(request):
    if request.method == "POST":
        url = request.POST.get("reference_url", None)
        reference_meta = request.POST.get("reference_meta", None)
        logger.debug("downloading {0} with meta {1}".format(url, reference_meta))
        if url is not None:
            reference_args = json.loads(base64.b64decode(reference_meta))
            reference = ReferenceGenome(**reference_args)
            reference.save()
            start_reference_download(url, reference.id)
        return HttpResponseRedirect(urlresolvers.reverse("references_genome_download"))

    references = get_references() or []
    downloads = DownloadMonitor.objects.filter(tags__contains="reference").order_by('-created')
    for download in downloads:
        if download.size:
            download.percent_progress = "{0:.2%}".format(float(download.progress) / download.size)
        else:
            download.percent_progress = "..."
    ctx = {
        'downloads': downloads,
        'references': references
    }
    return render_to_response("rundb/configure/reference_download.html", ctx,
        context_instance=RequestContext(request))

def start_reference_download(url, reference_id):
    url = url
    monitor = DownloadMonitor(url=url, tags="reference")
    monitor.save()
    try:
        t = tasks.download_something.apply_async(
            (url, monitor.id),
            link=tasks.install_reference.subtask((reference_id,)))
    except Exception as err:
        monitor.status = "System Error: " + str(err)
        monitor.save()
