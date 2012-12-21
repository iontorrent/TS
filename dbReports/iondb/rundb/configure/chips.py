#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''Functions supporting Instrument Diagnostics Page.  Primarily AutoPH files
transferred from PGM instrument'''

import os
import tarfile
import fnmatch
import logging
import traceback
import subprocess
from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.template import RequestContext

from iondb.rundb.models import GlobalConfig, FileServer


def showpage(request):
    '''Main Chips Files Display'''
    site_name = GlobalConfig.get().site_name

    # search all File Servers for a "Chips" directory
    fileservers = FileServer.objects.all()
    files = {}
    locList = []
    for server in fileservers:
        directory = os.path.join(server.filesPrefix, 'Chips')
        if os.path.isdir(directory):
            files[server.name] = []
            listoffiles = os.listdir(directory)
            listoffiles.sort()
            listoffiles.reverse()
            for filename in listoffiles:
                if fnmatch.fnmatch(filename, "*AutoPH*.bz2"):
                    #instName = string.split(filename, '_')[0]
                    instName = filename.split('_')[0]
                    if not [instName, server.name] in locList:
                        locList.append([instName, server.name])

                    if fnmatch.fnmatch(filename, "*AutoPHFail*"):
                        passVar = 'F'
                    elif fnmatch.fnmatch(filename, "*AutoPHPass*"):
                        passVar = 'T'

                    files[server.name].append([filename.split('.')[0], instName, os.path.join(directory, filename), passVar])

    ctxd = {
        "error_state": 0,
        "locations_list": locList,
        "base_site_name": site_name,
        "files": files,
    }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/ion_chips.html", context_instance=ctx)


def getChipZip(request, path):
    '''Download the AutoPH file, converted to zip compression'''
    from django.core.servers.basehttp import FileWrapper
    import zipfile
    logger = logging.getLogger(__name__)
    path = os.path.join("/", path)
    try:
        name = os.path.basename(path)
        name = name.split(".")[0]

        # initialize zip archive file
        zipfilename = os.path.join("/tmp", "%s.zip" % name)
        zipobj = zipfile.ZipFile(zipfilename, mode='w', allowZip64=True)

        # open tar.bz2 file, extract all members and write to zip archive
        tf = tarfile.open(os.path.join(path))
        for tarobj in tf.getmembers():
            contents = tf.extractfile(tarobj)
            zipobj.writestr(tarobj.name, contents.read())
        zipobj.close()

        response = HttpResponse(FileWrapper(open(zipfilename)), mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % os.path.basename(zipfilename)
        os.unlink(zipfilename)
        return response
    except Exception as inst:
        logger.exception(traceback.format_exc())
        ctxd = {
            "error_state": 1,
            "error": [['Error', '%s' % inst], ['Error type', '%s' % type(inst)]],
            "locations_list": [],
            "base_site_name": 'Error',
            "files": [],
        }
        ctx = RequestContext(request, ctxd)
        return render_to_response("rundb/configure/ion_chips.html", context_instance=ctx)


def getChipLog(request, path):
    '''Display InitLog.log contents'''
    logger = logging.getLogger(__name__)
    path = os.path.join("/", path)
    logLines = []
    try:
        tf = tarfile.open('%s' % path)
        ti = tf.extractfile('InitLog.txt')
        for line in ti.readlines():
            logLines.append(line)
        tf.close()
    except:
        logLines.append(traceback.format_exc())
        logger.exception(traceback.format_exc())

    ctxd = {"lineList": logLines}
    ctx = RequestContext(request, ctxd)
    return render_to_response('rundb/configure/ion_chipLog.html', context_instance=ctx)


def getChipPdf(request, path):
    '''Download Report document in PDF format'''
    import re
    from django.core.servers.basehttp import FileWrapper

    def runFromShell(cmd1):
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p1.communicate()
        return p1
    
    path = os.path.join("/", path)

    # File I/O setup
    tmpstem = os.path.basename(path).split('.')[0]
    tmphtml = os.path.join('/tmp', tmpstem + '.html')
    tmppdf = os.path.join('/tmp', tmpstem + '.pdf')
    tf = tarfile.open(path)
    ti = tf.extractfile('InitLog.txt')

    # regular expressions for the string parsing
    ph = re.compile(r'\d*\)\sW2\spH=\d*.\d*')
    phpass = re.compile(r'(W2\sCalibrate\sPassed\sPH=)(\d*.\d*)')
    adding = re.compile(r'(\d*\)\sAdding\s)(\d*.\d*)(\sml)')
    datefilter = re.compile(r'Sun|Mon|Tue|Wed|Thu|Fri|Sat')  # Simple filter for the line with the date on it.
    namefilter = re.compile(r'(Name:\s*)([a-z][a-z0-9_]*)', re.IGNORECASE)
    serialfilter = re.compile(r'(Serial)(.*?)(:)(\s*)([a-z][a-z0-9_]*)', re.IGNORECASE)
    surfacefilter = re.compile(r'(surface)(=)((?:[a-z][a-z]+))(\\s+)', re.IGNORECASE)
    rawtracefilter = re.compile(r'(RawTraces)(\s+)((?:[a-z][a-z]*[0-9]+[a-z0-9]*))(:)(\s+)([+-]?\d*\.\d+)(?![-+0-9\.])', re.IGNORECASE)

    # initialize variables
    initialph = ''
    finalph = ''
    totalAdded = 0.0
    iterationBuffer = ''
    rawtraceBuffer = ''
    totalIterations = 0
    startdate = ''
    enddate = ''
    pgmname = ''
    serialnumber = ''
    calstatus = ''
    surface = ''
    rawtracestartdate = ''

    # Log file parsing
    for line in ti.readlines():
        test = namefilter.match(line)
        if test:
            pgmname = test.group(2)

        test = serialfilter.match(line)
        if test:
            serialnumber = test.group(5)

        test = ph.match(line)
        if test:
            if initialph == '':
                initialph = test.group().split('=')[1]
            iterationBuffer += "<tr><td>%s</td></tr>\n" % test.group()
            totalIterations += 1

        test = adding.match(line)
        if test:
            iterationBuffer += "<tr><td>%s</td></tr>\n" % test.group()
            totalAdded += float(test.group(2))

        test = phpass.match(line)
        if test:
            finalph = test.group(2)
            calstatus = 'PASSED'

        if datefilter.match(line):
            if startdate == '':
                startdate = line.strip()

        test = surfacefilter.match(line)
        if test:
            surface = test.group(3)

        test = rawtracefilter.match(line)
        if test:
            rawtraceBuffer += "<tr><td>%s %s: %s</td></tr>\n" % (test.group(1), test.group(3), test.group(6))

    # Find the end date of the Chip Calibration - we need multilines to identify the end date entry
    # We are assuming that line endings are always newline char.
    ti.seek(0)
    contents = ti.read()
    enddatefilter = re.compile('^(W2 Calibrate Passed.*$\n)(Added.*$\n)([Sun|Mon|Tue|Wed|Thu|Fri|Sat].*$)', re.MULTILINE)
    m = enddatefilter.search(contents, re.MULTILINE)
    if m:
        enddate = m.group(3)

    startrawfilter = re.compile('([Sun|Mon|Tue|Wed|Thu|Fri|Sat].*$\n)(RawTraces.*$)', re.MULTILINE)
    m = startrawfilter.search(contents, re.MULTILINE)
    if m:
        rawtracestartdate = m.group(1)
        rawtraceBuffer = ('<tr><td>Raw Traces</td><td></td><td>%s</td></tr>' % rawtracestartdate) + rawtraceBuffer

    tf.close()

    f = open(tmphtml, 'w')
    f.write('<html>\n')
    f.write("<img src='/var/www/site_media/images/logo_top_right_banner.png' alt='lifetechnologies, inc.'/>")
    # If there are sufficient errors in parsing, display an error banner
    if calstatus == '' and finalph == '':
        f.write('<table width="100%">')
        f.write('<tr><td></td></tr>')
        f.write('<tr><td align=center><hr /><font color=red><i><h2>* * * Error parsing InitLog.txt * * *</h2></i></font><hr /></td></tr>')
        f.write('<tr><td></td></tr>')
        f.write("</table>")
    else:
        f.write('<table width="100%">')
        f.write('<tr><td></td></tr>')
        f.write('<tr><td align=center><hr /><i><h2>Instrument Installation Report</h2></i><hr /></td></tr>')
        f.write('<tr><td></td></tr>')
        f.write("</table>")

    f.write('<table width="100%">')
    f.write('<tr><td>Instrument Name</td><td>%s</td></tr>\n' % (pgmname))
    f.write('<tr><td>Serial Number</td><td>%s</td></tr>\n' % (serialnumber))
    f.write('<tr><td>Chip Surface</td><td>%s</td></tr>\n' % (surface))
    f.write("<tr><td>Initial pH:</td><td>%s</td><td>%s</td></tr>\n" % (initialph, startdate))
    f.write("<tr><td></td></tr>\n")  # puts a small line space
    f.write(iterationBuffer)
    f.write("<tr><td></td></tr>\n")  # puts a small line space
    f.write('<tr><td>Total Hydroxide Added:</td><td>%0.2f ml</td></tr>\n' % totalAdded)
    f.write('<tr><td>Total Iterations:</td><td>%d</td></tr>\n' % totalIterations)
    f.write("<tr><td>Final pH:</td><td>%s</td><td>%s</td></tr>\n" % (finalph, enddate))
    f.write("<tr><td></td></tr>\n")  # puts a small line space
    f.write(rawtraceBuffer)
    f.write("<tr><td></td></tr>\n")  # puts a small line space
    f.write('<tr><td>Instrument Installation Status:</td><td><i><font color="#00aa00">%s</font></i></td></tr>\n' % calstatus)
    f.write("</table>")

    f.write('<br /><br />\n')
    f.write('<table frame="box">\n')
    f.write('<tr><th align="left">Acknowledged by:</th></tr>')
    f.write('<tr><td align="left"><br />\n')
    f.write('__________________________________________________________<br />Customer Signature</td>')
    f.write('<td align="left"><br />___________________<br />Date\n')
    f.write('</td></tr>')
    f.write('<tr><td align="left"><br />\n')
    f.write('__________________________________________________________<br />Life Tech FSE Signature</td>')
    f.write('<td align="left"><br />___________________<br />Date\n')
    f.write('</td></tr></table></html>\n')
    f.close()
    pdf_cmd = ['/opt/ion/iondb/bin/wkhtmltopdf-amd64', str(tmphtml), str(tmppdf)]
    runFromShell(pdf_cmd)
    os.unlink(tmphtml)
    response = HttpResponse(FileWrapper(open(tmppdf)), mimetype='application/pdf')
    response['Content-Disposition'] = 'attachment; filename=%s' % (os.path.basename(tmppdf))
    # Can we delete the pdf file now?  Yes,on my dev box...
    os.unlink(tmppdf)

    return response
