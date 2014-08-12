#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
# vim: tabstop=4 shiftwidth=4 softtabstop=4 noexpandtab
# Ion Plugin - Ion Variant Caller

import os
import time
import json
import sys
import subprocess
import zipfile
import glob
import sqlite3

def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%X') + " ] " + message
    sys.stdout.flush()
    sys.stderr.flush()


def run_command(command,description):
    printtime(' ')
    printtime('Task    : ' + description)
    printtime('Command : ' + command)
    printtime(' ')
    return subprocess.call(command,shell=True)


WINDOW = 300

def _execute(path, query, limit, offset):
    """Function to execute queries against a local sqlite database"""

    page_str = " LIMIT {limit} OFFSET {offset}".format(limit=limit, offset=offset)

    if os.path.exists(os.path.join(path,"alleles.db")):
        dbPath = os.path.join(path,'alleles.db')
        connection = sqlite3.connect(dbPath)

        cursorobj = connection.cursor()
        try:
                cursorobj.execute(query + page_str)
                result = cursorobj.fetchall()

                #now find the total number of records that match the query
                cursorobj.execute(query.replace("SELECT *", "SELECT COUNT(*)") )
                count =  cursorobj.fetchall()
        except Exception as e:
                raise e
        connection.close()
        return count, result
    else:
        raise Exception("Unable to open database file")


def query(bucket):
    """returns a list of rows from database"""

    where = bucket["request_get"].get("where", False)
    #get the search filter JSON, then use that to build the SQL where statements
    where_list = []
    if where:
        where = json.loads(where)
        for key,value in where.iteritems():

                if isinstance(value,list):
                    for i,v in enumerate(value):
                        if i == 0:
                            where_local = "("
                        where_local += '"{key}" = "{value}"'.format(key=key,value=v)
                        if i+1 < len(value):
                            where_local += ' OR '
                        else:
                            where_local += ")"
                    where_list.append(where_local)
                elif key in ["Position", "Frequency", "Coverage"]:
                    where_list.append('"{key}" {value}'.format(key=key, value=value))
                elif key in ["Allele Name", "Gene ID", "Region Name"]:
                    where_list.append('"{key}" LIKE "%{value}%"'.format(key=key, value=value))

    if where_list:
        where_str = "WHERE "
        for i,w in enumerate(where_list):
            if i == 0:
                where_str += w
            else:
                where_str += " AND " + w
    else:
        where_str = ""

    limit = bucket["request_get"].get("limit", "20")
    offset = bucket["request_get"].get("offset", "0")
    #pass the path in from the JS, it is in startplugin.json - runinfo -> results_dir
    path = bucket["request_get"].get("path", "")

    #now do filtering 1 column and direction at a time
    column = bucket["request_get"].get("column", False)
    direction = bucket["request_get"].get("direction", "ASC")

    #TODO do the WHERE filtering
    #TODO just use string.format to make all the SQL, the built in parameterasation feature isn't good enough
    if column:
        q= 'SELECT * FROM variants {where} ORDER BY "{column}" {direction}'.format(
                        column=column, direction=direction, where=where_str)
        #lower case position is a special field that has a combination of chrm and the Position (int)
        if column == "position":
            #order by the by the ChromSort, then the position
            q= 'SELECT * FROM variants {where} ORDER BY "ChromSort" {direction}, "Position" {direction}'.format(
                direction=direction, where=where_str)
        count, rows = _execute(path, q, limit, offset)
    else:
        q = 'SELECT * FROM variants {where} ORDER BY "id"'.format(
                where=where_str)
        count, rows = _execute(path, q, limit, offset)

    #make the first item the total count of items
    data = {}
    data["total"] = [count[0][0]]
    data["items"] = []
    for row in rows:
        data["items"].append(row)
    return data


def start(path, plugin_path, temp_path, barcode):
    """
    Start an SGE job, and wait for it to return
    """

    #only import drmaa on the head nodes
    import drmaa

    #TODO: this is not robust and fails to init the session sometimes
    #code 11: Initialization failed due to existing DRMAA session.
    #AlreadyActiveSessionException: code 11: Initialization failed due to existing DRMAA session.

    os.environ["SGE_ROOT"] = "/var/lib/gridengine"
    os.environ["SGE_CELL"] = "iontorrent"
    os.environ["SGE_CLUSTER_NAME"] = "p6444"
    os.environ["SGE_QMASTER_PORT"] = "6444"
    os.environ["SGE_EXECD_PORT"] = "6445"
    os.environ["SGE_ENABLED"] = "True"
    os.environ["DRMAA_LIBRARY_PATH"] = "/usr/lib/libdrmaa.so.1.0"

    # create a single drmaa session
    _session = drmaa.Session()
    try:
        _session.initialize()
    except:
        _session.exit()
        #maybe need to return False?
        return False

    # Prepare drmaa Job - SGE/gridengine only
    jt = _session.createJobTemplate()
    jt.nativeSpecification = " -q %s" % ("plugin.q")

    jt.workingDirectory = path
    jt.outputPath = ":" + os.path.join(temp_path, "TVC_drmaa_stdout.txt")
    jt.joinFiles = True # Merge stdout and stderr

    jt.remoteCommand = "python"
    script = os.path.join(plugin_path, "extend.py")
    jt.args = [script, path, temp_path, barcode]

    # Submit the job to drmaa
    jobid = _session.runJob(jt)

    _session.deleteJobTemplate(jt)
    _session.exit()
    return jobid

def split(bucket):
    if "request_post" in bucket:
        #On the POST request do the splicing

        post = bucket["request_post"]

        #step one find out if this is a barcoded run http://ionwest.itw/report/39313/metal/barcodeList.txt

        path = post.get("startplugin", {}).get("runinfo", {}).get("results_dir", "")

        plugin_path = post.get("startplugin", {}).get("runinfo", {}).get("plugin_dir", "")

        variants = post.get("variants", {})

        barcode = post.get("barcode", False)

        #if it is a barcode make the dir inside of the barcodes dir
        if barcode:
            path = os.path.join(path, barcode)

        #TODO This don't show previously generated data
        if path:
            #Create a temp path to shuffle the data around inside of
            temp_path = "split_" + str(int(time.time()))

            #make the temp dir is it is there
            if not os.path.exists(os.path.join(path, temp_path)):
                os.mkdir(os.path.join(path, temp_path))

            #Write out the variants to a JSON file
            temp_json = os.path.join( path, temp_path, "variants.json")

            with open(temp_json, "w+") as temp:
                temp.write(json.dumps(variants))

            with open(os.path.join(path, "split_status.json" ), "w+") as status:
                status.write(json.dumps({"split_status":"Queued"}))

            job = start(path, plugin_path, temp_path, barcode)

            #TODO if two jobs are sent in, the UI is stupid and will think the first one done is the one it wants
            #TODO add the job id in the status as well, and check the right job is done.

            if job:
                #TODO : write the job id to the dir, just to try to allow one than on split to happen at once.
                return {"started" : temp_path , "job" : job }
            else:
                with open(os.path.join(path, "split_status.json" ), "w+") as status:
                    status.write(json.dumps({"split_status":"failed"}))
                return {"failed" : "Job not started SGE error"}
        else:
            return {"failed" : "Bad data POSTed"}

    else:
        return {"failed" : "This expects a POST request"}

def vcf_split(input_vcf, output_vcf, chr, position, window=300):

    #todo check boundaries later
    left = int(position) - (window / 2 )
    right = int(position) + (window / 2 )

    return ["vcftools", "--vcf", input_vcf, "--out", output_vcf, "--recode", "--keep-INFO-all", "--chr", chr,
            "--from-bp", str(left) , "--to-bp", str(right)]

def bam_split(input_bam, output_bam, chr, position, window=300):

    #todo check boundaries later
    left = int(position) - (window / 2 )
    right = int(position) + (window / 2 )
    location = chr + ":" + str(left) + "-" + str(right)

    return ["samtools", "view", "-h", "-b", input_bam, location, "-o", output_bam]

def bam_index(input_bam):
    """make the bam index"""
    return ["samtools", "index", input_bam]

def status_update(path, status):
    with open(os.path.join(path, "split_status.json"), "w+") as f:
        f.write(json.dumps({"split_status": str(status) }))

def clean_position_name(chrom, pos, position_names, window, id):
    """try to clean up chr names"""

    left = int(pos) - (window / 2 )
    right = int(pos) + (window / 2 )

    name = chrom.replace("|","PIPE")
    name += "_" + str(pos)
    name += "_padding_" + str(window)

    #now check to make sure that the name isn't already taken
    if name in position_names:
        name += "_" + str(id)

    #maybe like this later
    #chr#(start-padding)(end+padding) where end=start+length(ref).

    return name

if __name__ == '__main__':
    """extend.py main will do the data slicing, and spit out a zip"""

    printtime('Slicing started using: ' + (' '.join(sys.argv)))

    #where the run data lives
    path = sys.argv[1]
    #name of the working dir used for the temp files
    temp_path = sys.argv[2]
    try:
        barcode = sys.argv[3]
    except IndexError:
        barcode = False

    full_path = os.path.join(path, temp_path)

    rawlib = "rawlib.bam"
    prefix = ""

    #is there a ptrim file?
    ptrim = ""
    source_bed_files = []

    if not barcode:
        startplugin = json.load(open(os.path.join(path, "startplugin.json")))
        ptrim = os.path.join(path, "rawlib_processed.bam")
        source_bed_files = glob.glob(path+'/*.bed')
    else:
        #load it from the parent dir if it is a barcode
        startplugin = json.load(open(os.path.join(path, "../startplugin.json")))
        prefix = barcode + "_"
        rawlib = prefix + rawlib
        source_bed_files = glob.glob(path+'/../*.bed')

        try:
            ptrim = glob.glob(path +  "/*_processed.bam")[0]
        except IndexError:
            ptrim = ""

    printtime(str(source_bed_files))

    results_name = startplugin.get("expmeta", {}).get("results_name","results")
    reference = startplugin.get("runinfo", {}).get("library","hg19")
    variants = json.load(open(os.path.join(full_path, "variants.json")))

    status_update(path, "Stat Generation in progress")
    printtime("Generating BAM stats")

    #TODO BAM stats should be made by the plugin

    #TODO check to see if it exists
    #TODO if not then make it in the parent dir then copy it over

    #if it is there just added it to the zip later
    #Get full bam stats
    rawlib_stats = subprocess.Popen(['samtools', 'flagstat', os.path.join(path, rawlib)],
                                    stdout=subprocess.PIPE).communicate()[0]

    with open(os.path.join(full_path, prefix + "rawlib_stats.txt"),"w+") as f:
        f.write(rawlib_stats)

    if ptrim:
        rawlib_prtim_stats = subprocess.Popen(['samtools', 'flagstat', ptrim],
                                              stdout=subprocess.PIPE).communicate()[0]

        with open(os.path.join(full_path, prefix +  "rawlib_ptrim_stats.txt"),"w+") as f:
            f.write(rawlib_prtim_stats)

    #TODO add the bed files

    vcf_files = ["small_variants.vcf", "small_variants_filtered.vcf", "TSVC_variants.vcf"]

    status_update(path, "BAM and VCF files are being sliced")
    printtime("Generating expected VCF files")

    to_zip = []

    #write hotspot variants to force eval

    #TODO write one at a time
    for k,v in variants.iteritems():
        expected_bed_filename = prefix + str(v["chrom"]) + "_" + str(v["pos"]) + "_" + "expected.bed"
        expected_vcf_filename = prefix + str(v["chrom"]) + "_" + str(v["pos"]) + "_" + "expected.vcf"
        
        with open(os.path.join(full_path, expected_bed_filename),"w+") as f:
            pos = int(v.get("pos",0))
            ref = str(v.get("ref",""))
            f.write('track type=bedDetail\n')
            f.write(str(v.get("chrom",".")) + "\t" +
                    str(pos-1) + "\t" +
                    str(pos-1+len(ref)) + "\t" +
                    "." + "\t" +
                    "REF=" + ref + ";OBS=" + str(v.get("expected","")) + "\t" +
                    "." + "\n"
            )
        to_zip.append((os.path.join(full_path, expected_bed_filename), expected_vcf_filename))
        
        args =  "tvcutils prepare_hotspots"
        args += " --input-bed " + expected_bed_filename
        args += " --output-vcf " + expected_vcf_filename
        args += " --reference /results/referenceLibrary/tmap-f3/"+reference+"/"+reference+".fasta"
        args += " --left-alignment on"
        printtime(args)
        subprocess.check_call(args, cwd=full_path, shell=True)
        to_zip.append((os.path.join(full_path, expected_vcf_filename), expected_vcf_filename))
        

    #store all the names so that we don't use the same one twice
    position_names = []
    #Do splicing for each of the variants
    for id, variant in variants.iteritems():
        
        #base filename
        base = clean_position_name(variant["chrom"], variant["pos"], position_names, WINDOW, id)
        position_names.append(base)

        printtime("Slicing BAM and VCF files for " + base)

        #rawlib bam
        rawlib_variant = prefix + "rawlib_" + base + ".bam"
        args = bam_split(os.path.join(path, rawlib), rawlib_variant, variant["chrom"], variant["pos"], WINDOW)
        subprocess.check_call(args, cwd=full_path)
        to_zip.append((os.path.join(full_path, rawlib_variant), rawlib_variant))

        #make indexes
        args = bam_index(os.path.join(full_path, rawlib_variant))
        subprocess.check_call(args, cwd=full_path)
        to_zip.append((os.path.join(full_path, rawlib_variant + ".bai"), rawlib_variant + ".bai"))

        if ptrim:
            rawlib_ptrim_variant = prefix + "rawlib_processed_" + base + ".bam"
            args = bam_split(ptrim, rawlib_ptrim_variant, variant["chrom"], variant["pos"], WINDOW)
            subprocess.check_call(args, cwd=full_path)
            to_zip.append((os.path.join(full_path, rawlib_ptrim_variant), rawlib_ptrim_variant))

            #make indexes
            args = bam_index(os.path.join(full_path, rawlib_ptrim_variant))
            subprocess.check_call(args, cwd=full_path)
            to_zip.append((os.path.join(full_path, rawlib_ptrim_variant) + ".bai", rawlib_ptrim_variant + ".bai"))

        #make the tiny vcfs
        for vcf in vcf_files:
            vcf_for_variant = vcf.split(".")[0] + "_" + base
            args = vcf_split(os.path.join(path, vcf), vcf_for_variant, variant["chrom"], variant["pos"], WINDOW)
            #I don't think the error code that is returned can be trusted.
            vcf_status = subprocess.check_call(args, cwd=full_path)
            to_zip.append((os.path.join(full_path, vcf_for_variant + ".recode.vcf"), vcf_for_variant + ".recode.vcf"))
        
        for bed in source_bed_files:
            bed_for_variant = os.path.join(full_path,os.path.basename(bed)[:-4] + "_" + base + '.bed')
            printtime("Converting " + bed + " to " + bed_for_variant)
            with open(bed,"r") as i:
                with open(bed_for_variant,"w") as o:
                    o.write(i.readline())
                    for line in i:
                        fields = line.strip().split('\t')
                        if len(fields) < 3:
                            continue
                        startpos = int(fields[1])
                        endpos = int(fields[2])
                        if fields[0] != variant["chrom"]:
                            continue
                        if startpos > int(variant["pos"]) + (WINDOW/2):
                            continue
                        if endpos < int(variant["pos"]) - (WINDOW/2):
                            continue
                        o.write(line)
            to_zip.append((bed_for_variant, os.path.basename(bed_for_variant)))
            
        

    zip = zipfile.ZipFile(os.path.join(full_path,  results_name + ".zip"), "w")

    status_update(path, "Creating ZIP")
    printtime("Generating final ZIP file")
    missing_vcfs = []
    for source, dest in to_zip:
        try:
            zip.write(source, os.path.join(results_name, dest))
            #so if it got here then we now the vcf was there, we can know try to bgzip and index it.
            if source.endswith(".vcf"):
                #make the bgzip, same file name with .gz added to the end it also replaces the old file
                #yes it is redundant to include this file twice. but it should be tiny
                subprocess.check_call(["bgzip", source], cwd=full_path)
                zip.write(source + ".gz" , os.path.join(results_name, dest + ".gz"))
                subprocess.check_call(["tabix", "-f" , "-p", "vcf", source + ".gz"], cwd=full_path)
                zip.write(source + ".gz.tbi", os.path.join(results_name, dest + ".gz.tbi"))
        except OSError:
            missing_vcfs.append(source)

    #add bam stats to the zip
    zip.write(os.path.join(full_path, prefix + "rawlib_stats.txt"), os.path.join(results_name, prefix + "rawlib_stats.txt"))

    #if not barcode:
    #    zip.write(os.path.join(full_path,"rawlib_ptrim_stats.txt"), os.path.join(results_name, "rawlib_ptrim_stats.txt"))

    #parameters
    if not barcode:
        zip.write(os.path.join(path, "local_parameters.json"), os.path.join(results_name, "local_parameters.json"))
    else:
        zip.write(os.path.join(path, "../local_parameters.json"), os.path.join(results_name, "local_parameters.json"))

    #add the variants.json file
    zip.write(os.path.join(full_path, "variants.json"), os.path.join(results_name, "variants.json"))

    #if there are missing vcfs add those to the zip
    if missing_vcfs:
        with open(os.path.join(full_path, "missing_vcf.txt"), "w+") as f:
            for missing_vcf in missing_vcfs:
                f.write(missing_vcf + "\n")

        zip.write(os.path.join(full_path, "missing_vcf.txt"), os.path.join(results_name, "missing_vcf.txt"))

    zip.close()

    #Now that everything is done write out the status file which will be checked by the VC JavaScript
    with open(os.path.join(path, "split_status.json"), "w+") as f:
        f.write(json.dumps({"split_status": "done", "url" : os.path.join(temp_path, results_name + ".zip")}))

