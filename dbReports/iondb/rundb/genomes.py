# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import datetime
import os
import shutil
import socket
import xmlrpclib
import glob
import fileinput
import logging

logger = logging.getLogger(__name__)

import json

from ajax import render_to_json

from django import http, shortcuts, template
from django.conf import settings
from django.core import urlresolvers
from django import http

from iondb.anaserve import client
from iondb.rundb import forms
from iondb.rundb import models
from iondb.rundb import tasks


from django.conf import settings

def fileUpload(request):
    """file upload for plupload"""
    if request.method == 'POST':
        name = request.REQUEST.get('name','')
        uploaded_file = request.FILES['file']
        if not name:
            name = uploaded_file.name
        name,ext = os.path.splitext(name)

        #check to see if a user has uploaded a file before, and if they have
        #not, make them a upload directory

        upload_dir = "/results/referenceLibrary/temp/"

        if not os.path.exists(upload_dir):
            return render_to_json({"error":"upload path does not exist"})

        dest_path = '%s%s%s%s' % (upload_dir,os.sep,name,ext)

        chunk = request.REQUEST.get('chunk','0')
        chunks = request.REQUEST.get('chunks','0')

        debug = [chunk, chunks]

        with open(dest_path,('wb' if chunk==0 else 'ab')) as f:
            for content in uploaded_file.chunks():
                f.write(content)

        if int(chunk) + 1 >= int(chunks):
            #the upload has finished
            pass

        return render_to_json({"chuck posted":debug})

    else:
        return render_to_json({"method":"only post here"})


def references(request):
    """Display the test fragment templates and reference genomes stored in the database.
     there will be two secions of references
     1) the most current version
     2) everything else.
     """
    #look for new genomes
    search_for_genomes()
    rg, not_current_rg = [], []
    for reference in  models.ReferenceGenome.objects.all().order_by('short_name'):
        if reference.index_version == settings.TMAP_VERSION or reference.status == "Rebuilding index":
            rg.append(reference)
        else:
            not_current_rg.append(reference)
    tf = models.Template.objects.all().order_by('pk')
    barcodes = models.dnaBarcode.objects.values('name').distinct().order_by('name')
    
    ctx = template.RequestContext(request, {"templates":tf, "genomes": rg,
                                            "not_current_genomes" : not_current_rg,
                                            "barcodes": barcodes})
    
    return shortcuts.render_to_response("rundb/ion_references.html",
                                        context_instance=ctx)


def delete_genome(request, pk):
    """delete a reference genome
    the filesystem file deletions should be done with a method on the model"""

    if request.method=="POST":
        rg = shortcuts.get_object_or_404(models.ReferenceGenome, pk=pk)

        with_dir = request.GET.get('with_dir', False)

        #delete dir by default
        try_delete = rg.delete()

        if not try_delete:
            #the file could not be deleted, present the user with an error message.
            return render_to_json({"status":" <strong>Error</strong> <p>Genome could not be deleted.</p> \
                                          <p>Check the file permissions for the genome on the file system at: </p> \
                                          <p><strong>" + str(rg.reference_path) + "</p></strong> "})

        return render_to_json({"status":"Genome was deleted successfully"})

    if request.method == "GET":
        return render_to_json({"status":"This must be accessed via post"})


def change_genome_name(rg, new_name, old_full_name, new_full_name):
    """
    to change a genome name, we have to
    1) move the dir from the old name to the new one
    2) change the data in <genome_name>.info.txt
    3) change all of the files names to have a prefix of the new genome name
    4) rewrite the reference_list.txt files, which is being done from the calling function
    """

    if (new_name != rg.short_name):
        #we also really need to check to see if the file exsits.
        old_files = glob.glob( rg.reference_path + "/" + rg.short_name +".*")

        def rreplace(s, old, new, occurrence):
            """replace from starting at the right"""
            li = s.rsplit(old, occurrence)
            return new.join(li)

        for old_file in old_files:
            os.rename ( old_file, rreplace(old_file, rg.short_name,new_name, 1) )

        shutil.move( rg.reference_path , settings.TMAP_DIR + new_name )

    info = os.path.join(settings.TMAP_DIR,new_name,new_name + ".info.txt")

    #this will rewrite the genome.info.text file
    for line in fileinput.input(info,inplace=1):
        if line.split('\t')[0] == "genome_name":
            print line.replace(old_full_name, new_full_name),
        else:
            print line,

def genome_set_alignment_sample(pk,size_to_set):
    """Allow the alignment sample size to be changed"""

    try:
        rg = models.ReferenceGenome.objects.get(pk=pk)
        size_to_set = int(size_to_set)
    except:
        return False

    genome_info = read_genome_info(rg.info_text())

    if size_to_set == 0:
        try:
            del genome_info['read_sample_size']
        except KeyError:
            return
        
        write_genome_info(rg.info_text(),genome_info)
        return

    if size_to_set > 0:
        genome_info["read_sample_size"] = size_to_set
        write_genome_info(rg.info_text(),genome_info)
        return


def write_genome_info(info_path,dict):
    """write genome info to file from dict
    """
    try:
        genome_info = open(info_path,'w')
        for key,value in dict.items():
            genome_info.write(str(key))
            genome_info.write("\t")
            genome_info.write(str(value))
            genome_info.write("\n")

    except IOError:
        return False

    genome_info.close()

    return True

def read_genome_info(info_path):
    """each genome has a genome.info.txt file which contains meta info for that genome
    here we will find and return that as a string
    if False is returned the genome can be considered broken
    """

    try:
        genome_info = open(info_path).readlines()

        #build a dict with the values from the info.txt
        genome_dict = {"genome_name": False, "genome_version" : False, "index_version" : False }
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

def genome_get_fasta(pk):
    """each genome should have a fasta file
        check if that exists, and if it does return a link
        we also have to provide someway to download the fasta files with apache
    """

    try:
        rg = models.ReferenceGenome.objects.get(pk=pk)
    except:
        return False

    genome_fasta = os.path.join(rg.reference_path , rg.short_name +".fasta")

    size = False
    
    try:
        os.path.exists(genome_fasta)
        size = os.path.getsize(genome_fasta)
    except IOError:
        genome_fasta = False
    except os.error:
        genome_fasta = False

    return genome_fasta, size

def verbose_error_trim(verbose_error):
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
        

def edit_genome(request, pk):
    """Make changes to an existing genome database reference,
    or create a new one if ``pk`` is zero."""

    rg = shortcuts.get_object_or_404(models.ReferenceGenome, pk=pk)
    uploads = models.ContentUpload.objects.filter(publisher__name="BED")
    relevant = [u for u in uploads if u.meta.get("reference","") == rg.short_name]
    #TODO give an indication if it is a hotspot BED file
    bedFiles, processingBedFiles = [], []
    for upload in relevant:
        info = {"path": os.path.basename(upload.file_path), "pk": upload.pk}
        if upload.status == "Successfully Completed":
            bedFiles.append(info)
        else:
            info["status"] = upload.status
            processingBedFiles.append(info)

    if request.method=="POST":
        rfd = forms.EditReferenceGenome(request.POST)
        if rfd.is_valid():
            rg.notes = rfd.cleaned_data['notes']
            rg.enabled = rfd.cleaned_data['enabled']
            rg.date = datetime.datetime.now()

            if (rg.short_name != rfd.cleaned_data['name'] or rg.name !=  rfd.cleaned_data['NCBI_name']):
                change_genome_name(rg, rfd.cleaned_data['name'], rg.name, rfd.cleaned_data['NCBI_name'])

            #make sure to only set the new name after the change_genome_name call - it needs the old full name
            rg.name = rfd.cleaned_data['NCBI_name']
            rg.short_name = rfd.cleaned_data['name']

            #Update the reference path
            if rg.enabled:
                enabled_path = os.path.join(settings.TMAP_DIR, rg.short_name)
                rg.enable_genome()
            else:
                base_path = os.path.split(os.path.split(rg.reference_path)[0])[0]
                enabled_path= os.path.join(base_path, "disabled", rg.index_version, rg.short_name)
                rg.disable_genome()

            rg.reference_path = enabled_path

            rg.save()

            genome_set_alignment_sample(rg.pk,rfd.cleaned_data['read_sample_size'])
            
            url = urlresolvers.reverse("configure_references")
            return http.HttpResponsePermanentRedirect(url)
        else:
            genome_dict = read_genome_info(rg.info_text())
            verbose_error = verbose_error_trim(rg.verbose_error)
            genome_fasta, genome_size = genome_get_fasta(pk)

            ctxd = {"temp":rfd, "name":rg.short_name, "key" : rg.pk, "enabled" : rg.enabled,
                    "genome_dict" : genome_dict, "status": rg.status , "verbose_error" : verbose_error,
                    "genome_fasta" : genome_fasta, "genome_size" : genome_size,
                    "bedFiles" : bedFiles, "processingBedFiles": processingBedFiles,
                    "index_version" : rg.index_version
            }
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/configure/edit_reference.html",
                                                context_instance=ctx)
    elif request.method=="GET":
        temp = forms.EditReferenceGenome()
        temp.fields['NCBI_name'].initial = rg.name
        temp.fields['name'].initial = rg.short_name
        temp.fields['notes'].initial = rg.notes
        temp.fields['enabled'].initial = rg.enabled
        temp.fields['genome_key'].initial = pk
        temp.fields['index_version'].initial = rg.index_version

        genome_dict = read_genome_info(rg.info_text())

        try:
            sample_size = genome_dict["read_sample_size"]
            temp.fields['read_sample_size'].initial = sample_size
        except:
            temp.fields['read_sample_size'].initial = 0

        genome_fasta, genome_size = genome_get_fasta(pk)

        verbose_error = verbose_error_trim(rg.verbose_error)
        fastaOrig = rg.fastaOrig()

        stale_index = rg.index_version != settings.TMAP_VERSION and rg.status != "Rebuilding index"

        ctxd = {"temp":temp, "name":rg.short_name, "key" : rg.pk, "enabled" : rg.enabled,
                "genome_dict" : genome_dict, "status": rg.status , "verbose_error" : verbose_error,
                "genome_fasta" : genome_fasta, "genome_size" : genome_size,
                "index_version" : rg.index_version, "fastaOrig" : fastaOrig,
                "bedFiles": bedFiles, "processingBedFiles": processingBedFiles,
                "stale_index": stale_index
        }
        ctx = template.RequestContext(request, ctxd)
        return shortcuts.render_to_response("rundb/configure/edit_reference.html",
                                            context_instance=ctx)

def genome_status(request, pk):
    """Provide a way for the index creator to let us know when the index has been created"""

    if request.method=="POST":
        rg = shortcuts.get_object_or_404(models.ReferenceGenome, pk=pk)
        status = request.POST.get('status',False)
        enabled = request.POST.get('enabled',False)
        verbose_error = request.POST.get('verbose_error',"")
        index_version = request.POST.get('index_version',"")

        if not status:
            return render_to_json({"status":"error genome status not given"})

        rg.status  = status
        rg.enabled = enabled
        rg.verbose_error = verbose_error
        rg.index_version = index_version
        rg.reference_path = os.path.join(settings.TMAP_DIR, rg.short_name)

        rg.save()
        return render_to_json({"status":"genome status updated", "enabled" : enabled })
    if request.method=="GET":
        rg = shortcuts.get_object_or_404(models.ReferenceGenome, pk=pk)
        return render_to_json({"status":rg.status})


def search_for_genomes():
    """
    Searches for new genomes.  This will sync the file system and the genomes know by the database
    """

    ref_dir = '/results/referenceLibrary'

    lib_versions = []

    for folder in os.listdir(ref_dir):
        if os.path.isdir(os.path.join(ref_dir,folder)) and folder.lower().startswith("tmap"):
            lib_versions.append(folder)
    logger.debug("Reference genome scanner found %s" % ",".join(lib_versions))
    for lib_version in lib_versions:
        if os.path.exists(os.path.join(ref_dir,lib_version)):
            libs = os.listdir(os.path.join(ref_dir,lib_version))
            for lib in libs:
                genome_info_text = os.path.join(ref_dir, lib_version, lib , lib +".info.txt")
                logger.debug("Parsing reference genome info %s" % genome_info_text)
                genome_dict= read_genome_info(genome_info_text)
                #TODO: we have to take into account the genomes that are queue for creation of in creation

                if genome_dict:
                    #here we trust that the path the genome is in, is also the short name
                    existing_reference = models.ReferenceGenome.objects.filter(
                        short_name=lib).order_by("-index_version")[:1]
                    if existing_reference:
                        rg = existing_reference[0]
                        logger.debug("Found existing genome %s id=%d index=%s" % (
                            str(rg), rg.id, rg.index_version))
                        if rg.index_version != genome_dict["index_version"]:
                            logger.debug("Updating genome status to 'found'")
                            rg.status = "found"
                            try:
                                rg.name = genome_dict["genome_name"]
                                rg.version = genome_dict["genome_version"]
                                rg.index_version = genome_dict["index_version"]
                                rg.reference_path = os.path.join(ref_dir, rg.index_version, rg.short_name )
                            except:
                                rg.name = lib
                                rg.status = "missing info.txt"
                            rg.save()
                    else:
                        logger.debug("Found new genome %s index=%s" % (
                            lib, genome_dict["genome_version"]))
                        #the reference was not found, add it to the db
                        rg = models.ReferenceGenome()
                        rg.short_name = lib
                        rg.date = datetime.datetime.now()
                        rg.status = "found"
                        rg.enabled = True

                        rg.index_version = ""
                        rg.version = ""
                        rg.name = ""

                        try:
                            rg.name = genome_dict["genome_name"]
                            rg.version = genome_dict["genome_version"]
                            rg.index_version = genome_dict["index_version"]
                            rg.reference_path = os.path.join(ref_dir, rg.index_version, rg.short_name )
                        except:
                            rg.name = lib
                            rg.status = "missing info.txt"

                        rg.save()
                        logger.info("Created new reference genome %s id=%d" % (
                            str(rg), rg.id))


def new_genome(request):
    """This is the page to create a new genome. The XML-RPC server is ionJobServer.
    """

    if request.method=="POST":
        """parse the data sent in"""

        #required
        name = request.POST.get('name',False)
        short_name = request.POST.get('short_name',False)
        fasta = request.POST.get('target_file',False)
        version = request.POST.get('version',False)
        notes = request.POST.get('notes',"")

        #optional
        read_sample_size = request.POST.get('read_sample_size',False)
        read_exclude_length = request.POST.get('read_exclude_length',False)

        #URL download
        url = request.POST.get('url',False)

        #if any of those were false send back a failed message
        if not all((name, short_name, fasta, version)):
            return render_to_json({"status":"Form validation failed","error":True})

        if not set(short_name).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"):
            return render_to_json( {"status":"The short name has invalid characters. The valid values are letters, numbers, and underscores.","error": True} )

        #TODO: check to make sure the zip file only has one fasta or fa

        path = "/results/referenceLibrary/temp/"
        if not url:
            #check to ensure the size on the OS the same as the reported.
            reported_file_size = request.POST.get('reported_file_size',False)

            try:
                uploaded_file_size = str(os.path.getsize(path + fasta))
            except OSError:
                return render_to_json( {"status":"The FASTA temporary files was not found","error":True} )

            if reported_file_size != uploaded_file_size :
                try:
                    os.remove(path + fasta)
                    pass
                except OSError:
                    return render_to_json( {"status":"The FASTA temporary did not match the expected size, and could not be deleted.","error":True} )
                return render_to_json( {"status":
                                        "The file you uploaded differs from the expected size. This is due to an error uploading.  The temporary file has been removed.",
                                        "reported": reported_file_size,
                                        "uploaded": uploaded_file_size,
                                        "error" : True
                } )

        #Make an genome ref object
        if models.ReferenceGenome.objects.filter(short_name=short_name,index_version=settings.TMAP_VERSION):
            #check to see if the genome already exists in the database with the same version
            return render_to_json({"status":"Failed - Genome with this short name and index version already exist.","error":True})
        rg = models.ReferenceGenome()
        rg.name = name
        rg.short_name = short_name
        rg.version = version
        rg.date = datetime.datetime.now()
        rg.notes = notes
        rg.status = "queued"
        rg.enabled = False
        rg.index_version = settings.TMAP_VERSION

        #before the object is saved we should ping the xml-rpc server to see if it is alive.
        try:
            host = "127.0.0.1"
            conn = client.connect(host,settings.JOBSERVER_PORT)
            #just check uptime to make sure the call does not fail
            conn.uptime()
        except (socket.error,xmlrpclib.Fault):
            return render_to_json( {"status":"Unable to connect to ionJobserver process.  You may need to restart ionJobserver","error":True} )

        #if the above didn't fail then we can save the object
        #this object must be saved before the tmap call is made
        rg.save()

        #kick off the anaserve tmap xmlrpc call
        import traceback
        try:
            host = "127.0.0.1"
            conn = client.connect(host,settings.JOBSERVER_PORT)
            tmap_bool, tmap_status = conn.tmap(str(rg.id), fasta, short_name, name, version, read_sample_size, read_exclude_length, settings.TMAP_VERSION)
        except (socket.error,xmlrpclib.Fault):
            #delete the genome object, because it was not sucessful
            rg.delete()
            return render_to_json( {"status":"Error with index creation", "error" : traceback.format_exc() } )

        if not tmap_bool:
            rg.delete()
            return render_to_json( {"status":tmap_status,"error":True} )

        return render_to_json({"status":"The genome index is being created.  This might take a while, check the status on the references tab. You are being redirected there now.","error":False})

    elif request.method=="GET":

        ctxd = {}
        ctx = template.RequestContext(request, ctxd)

        #when we get a POST that data should be validated and go to the xmlrpc process
        return shortcuts.render_to_response("rundb/ion_new_genome.html", context_instance=ctx)


def rebuild_index(reference):
    """Add a job to rebuild the reference index for reference to the SGE queue
    """
    logger.info("Queuing TMAP reference index rebuild of %s" % reference.short_name)
    reference.status = "Rebuilding index"
    reference.save()
    tasks.build_tmap_index.delay(reference)


def start_index_rebuild(request, reference_id):
    data = {"references":[]}
    if reference_id == "all":
        references = models.ReferenceGenome.objects.exclude(index_version=settings.TMAP_VERSION)
        logger.info("Rebuilding TMAP reference indices for %s" %
                    ", ".join(r.short_name for r in references))
        for reference in references:
            rebuild_index(reference)
            data["references"].append({"id":reference.pk,
                                       "short_name": reference.short_name})
    else:
        reference = models.ReferenceGenome.objects.get(pk=reference_id)
        rebuild_index(reference)
        data["references"].append({"id":reference.pk,
                                   "short_name": reference.short_name})
    return http.HttpResponse(json.dumps(data) , mimetype="application/json")
