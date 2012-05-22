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

ampl_plugin = 'ampliconGeneralAnalysis'
    
def get_result_type(result,pstore):
  rtype = None
  if result.experiment.library.startswith('ampl_') or (ampl_plugin in pstore.keys() and pstore[ampl_plugin]):
    rtype = 'ampl'
#  elif 'merged' in result.experiment.project.lower(): #temporary by project name
#    rtype = 'paired'  
  elif result.resultsName[-3:] == '_tn':
    rtype = 'thumb'  
  else:
    rtype = 'full' #default type  
  return rtype      

def find_first_result(exp,pstore):
  rset = exp.results_set.all().order_by('timeStamp')
  firstPk = []
  firstType = []
  for i in range(0, len(rset), 1):
    testResult = rset[i]
    rtype = get_result_type(testResult,pstore)
    if (not firstType) or (rtype not in firstType):
        firstPk.append(testResult.pk)
        firstType.append(rtype)        
  return firstPk, firstType    

def send_nightly():
    # get the list of all results generated in the last 24 hours
    timerange = datetime.datetime.now() - datetime.timedelta(days=1)
    resultsList = models.Results.objects.filter(timeStamp__gt=timerange)
        
    resultsAll=[]
    rType=[]
    rNew=[]
    covAmpl=[]  
  
    for result in resultsList:
        if (result.status == 'Completed'):
            exp = result.experiment
            pstore = result.getPluginStore()            
            [firstPk, firstType] = find_first_result(exp,pstore)

            resultsAll.append(result)            
            rNew.append(result.pk in firstPk)
            rType.append(get_result_type(result,pstore))
            
            if (ampl_plugin in pstore.keys() ) and pstore[ampl_plugin]:
              covAmpl.append(pstore[ampl_plugin]['target_coverage_at_20x_-_norm_100'])
            else:
              covAmpl.append(0)
        
    #sort by chipType
    resultsAll, rType, rNew, covAmpl = zip(*sorted(zip(resultsAll,rType,rNew,covAmpl), key=lambda r:r[0].experiment.chipType))
    
    
    gc = models.GlobalConfig.objects.all()[0]
    web_root = gc.web_root
    if len(web_root) > 0:
        if web_root[-1] == '/':
            web_root = web_root[:len(web_root)-1]

    if len(gc.site_name) is 0:
        site_name = "<set site name in Global Configs on Admin Tab>"
    else:
        site_name = gc.site_name
     
    lbms = [res.best_lib_metrics for res in resultsAll]
    tfms = [res.best_metrics for res in resultsAll]
    links = [web_root+res.reportLink for res in resultsAll]
 
    #find the sum of the q17 bases
    hqBaseSum = 0
    for res,n,tp in zip(resultsAll,rNew,rType):
        if n and (not tp=='thumb') and (not tp=='paired') and res.best_lib_metrics:
            if not res.best_lib_metrics.align_sample == 0:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.q17_mapped_bases
            if res.best_lib_metrics.align_sample == 1:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.extrapolated_mapped_bases_in_q17_alignments
            if not res.best_lib_metrics.align_sample == 2:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.q17_mapped_bases
    
    tmpl = loader.get_template(TEMPLATE_NAME)
    ctx = template.Context({"reportsOld":
                                [(r,t,lb,l) for r,t,lb,l,tp,n in zip(resultsAll,tfms,lbms,links,rType,rNew) if not n],
                            "reportsNew":
                                [(r,t,lb,l) for r,t,lb,l,tp,n in zip(resultsAll,tfms,lbms,links,rType,rNew) if tp=='full' and n],
                            "reportsThumbsNew":
                                [(r,t,lb,l) for r,t,lb,l,tp,n in zip(resultsAll,tfms,lbms,links,rType,rNew) if tp=='thumb' and n],
                            "reportsAmplNew":
                                [(r,t,lb,l,c) for r,t,lb,l,tp,n,c in zip(resultsAll,tfms,lbms,links,rType,rNew,covAmpl) if tp=='ampl' and n],
                            "reportsPairNew":                              
                               [(r,t,lb,l) for r,t,lb,l,tp,n in zip(resultsAll,tfms,lbms,links,rType,rNew) if tp=='paired' and n],    
                            "webroot":web_root,
                            "sitename":site_name,
                            "hq_base_num_new":hqBaseSum,
                            "use_precontent":True,
                            "use_content2":True})
    html = tmpl.render(ctx)
    text = reports_to_text(resultsAll)
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

