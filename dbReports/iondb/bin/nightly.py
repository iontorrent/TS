#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *

import datetime


from django import template
from django.core import mail
from django.template import loader
from os import path
from iondb.rundb import models

settings.EMAIL_HOST = 'localhost'
settings.EMAIL_PORT = 25
settings.EMAIL_USE_TLS = False

def get_recips():
    emails = models.EmailAddress.objects.filter(selected=True)
    if len(emails) > 0:
        ret = [i.email for i in emails]
    else:
        sys.exit(0)
    return ret

RECIPS = get_recips()
#RECIPS = ['Mel.Davey@Lifetech.com']
SENDER = "donotreply@iontorrent.com"
TEMPLATE_NAME = "rundb/ion_nightly.html"

def reports_to_text(reports):
    return "Please enable HTML messages in your mail client."

def send_html(sender,recips,subject,html,text):
    msg = mail.EmailMessage(subject,html,sender,recips)
    msg.content_subtype = "html"
    msg.send()

def send_nightly():
    # get the list of all results generated in the last 24 hours
    timerange = datetime.datetime.now() - datetime.timedelta(days=1)
    resultsList = models.Results.objects.filter(timeStamp__gt=timerange)

    # for each result, its either the first result generated, or not.  If first, it goes into the new results list, else into the old
    # so first result is either the first thumbnail, or if no thumbnails, the first full report
    resultsNew = []
    resultsOld = []
    resultsThumbsNew = []

    for result in resultsList:
        if (result.status == 'Completed'):
            exp = result.experiment
            rset = exp.results_set.all().order_by('timeStamp')
            # now find the first thumbnail, if its the result we are looking at, its new
            # else find the first non-thumbnail, and if its the result we are looking at, then add to new,
            # else add to old
            firstThumb = None
            firstFull = None
            for i in range(0, len(rset), 1):
               testResult = rset[i]
               if (firstThumb is None) and (testResult.resultsName[-3:] == '_tn'):
                   firstThumb = testResult
               if (firstFull is None) and (testResult.resultsName[-3:] != '_tn'):
                   firstFull = testResult

            added = False
            if firstThumb is not None:
                if firstThumb.pk == result.pk:
                    resultsThumbsNew.append(result)
                    added = True
            # we could exclude adding the full report if we already added the thumb, but tricky since we would need a field to modify in each result indicating we just added it

            if not added and firstFull is not None:
                if firstFull.pk == result.pk:
                    resultsNew.append(result)
                    added = True

            if not added:
                resultsOld.append(result)

    #resultsOld = resultsOld.order_by('timeStamp')
    #resultsNew = resultsNew.order_by('timeStamp')
    
    gc = models.GlobalConfig.objects.all()[0]
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == '/':
            web_root = web_root[:len(web_root)-1]

    if len(gc.site_name) is 0:
        site_name = "<set site name in Global Configs on Admin Tab>"
    else:
        site_name = gc.site_name

    lbmsOld = [res.best_lib_metrics for res in resultsOld]
    tfmsOld = [res.best_metrics for res in resultsOld]
    linksOld = [web_root+res.reportLink for res in resultsOld]
    
    lbmsNew = [res.best_lib_metrics for res in resultsNew]
    tfmsNew = [res.best_metrics for res in resultsNew]
    linksNew = [web_root+res.reportLink for res in resultsNew]

    lbmsThumbsNew = [res.best_lib_metrics for res in resultsThumbsNew]
    tfmsThumbsNew = [res.best_metrics for res in resultsThumbsNew]
    linksThumbsNew = [web_root+res.reportLink for res in resultsThumbsNew]
 
    #find the sum of the q17 bases
    hqBaseSum = 0
    for res in resultsNew:
        if res.best_lib_metrics:
            if not res.best_lib_metrics.align_sample == 0:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.q17_mapped_bases
            if res.best_lib_metrics.align_sample == 1:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.extrapolated_mapped_bases_in_q17_alignments
            if not res.best_lib_metrics.align_sample == 2:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.q17_mapped_bases

    tmpl = loader.get_template(TEMPLATE_NAME)
    ctx = template.Context({"reportsOld":
                                [(r,t,lb,l) for r,t,lb,l in zip(resultsOld,tfmsOld,lbmsOld,linksOld) if t or l],
                            "reportsNew":
                                [(r,t,lb,l) for r,t,lb,l in zip(resultsNew,tfmsNew,lbmsNew,linksNew) if t or l],
                            "reportsThumbsNew":
                                [(r,t,lb,l) for r,t,lb,l in zip(resultsThumbsNew,tfmsThumbsNew,lbmsThumbsNew,linksThumbsNew) if t or l],
                            "webroot":web_root,
                            "sitename":site_name,
                            "hq_base_num_new":hqBaseSum,
                            "use_precontent":True,
                            "use_content2":True})
    html = tmpl.render(ctx)
    text = reports_to_text(resultsOld)
    subTitle = "[Report Summary] for %s %s, %s-%s-%s" % (site_name,"%a","%m","%d","%y")
    #subTitle = "[Report Summary] %a, %m-%d-%y"
    subject = datetime.datetime.now().strftime(subTitle)
    outfile = open('/tmp/out.html', 'w')
    outfile.write(html)
    outfile.close()
    return send_html(SENDER,RECIPS,subject,html,text)
    
def main(args):
    send_nightly()

if __name__ == '__main__':
    main(sys.argv)

