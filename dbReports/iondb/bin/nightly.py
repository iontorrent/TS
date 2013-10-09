#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *

import datetime
import pytz

from django import template
from django.core import mail
from django.template import loader
from os import path
from iondb.rundb import models
import ast

settings.EMAIL_HOST = 'localhost'
settings.EMAIL_PORT = 25
settings.EMAIL_USE_TLS = False

def get_recips():
    emails = models.EmailAddress.objects.filter(selected=True)
    return [i.email for i in emails]

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
  rtype = 'full'
  try:
    if "paired" in result.metaData.keys() and result.metaData["paired"] == 1:
      rtype = 'paired'
    elif result.eas.reference.startswith('ampl_') or (ampl_plugin in pstore.keys() and pstore[ampl_plugin]):
      rtype = 'ampl'
    elif "thumb" in result.metaData.keys() and result.metaData["thumb"] == 1:
      rtype = 'thumb'
    else:
      rtype = 'full' #default type  
  except:
    # problems with eas were likely, leets see if it was at least thumb or full
    if "thumb" in result.metaData.keys() and result.metaData["thumb"] == 1:
      rtype = 'thumb'

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
  
def paired_end_stats(pe_path):
# get metrics that PE report doesn't upload
  try:
    filenames = ['corrected.alignment.summary','fwd_alignment.summary','Paired_Fwd.alignment.summary'] 
    nreads = [0]*len(filenames)  
    for i,sfile in enumerate(filenames):
      with open(os.path.join(pe_path,sfile)) as f:
        for line in f.readlines():
          if "Total number of Reads" in line:
            nreads[i] = float(line.split('=')[1])
            break
    return 100*(nreads[0]+nreads[2])/nreads[1], 100*nreads[0]/nreads[1]        
  except:
    return 0,0  

def send_nightly():
    # get the list of all results generated in the last 24 hours
    timerange = datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=1)
    resultsList = models.Results.objects.filter(timeStamp__gt=timerange)
        
    resultsAll=[]
    rType=[]
    rNew=[]
    info=[]
  
    for result in resultsList:
        if (result.status == 'Completed' and 'INJECTED' not in result.resultsName):
            exp = result.experiment
            pstore = result.getPluginStore()            
            [firstPk, firstType] = find_first_result(exp,pstore)

            resultsAll.append(result)            
            rNew.append(result.pk in firstPk)
            rType.append(get_result_type(result,pstore))
 
            # Here's a bunch of random other things we want to track, all stuffed into this extraInfo dict array
            extraInfo = {}

            # pull the Q20 estimated bases from the quality metrics table
            try:
                qmetrics = result.qualitymetrics_set.all()[0]
                extraInfo["q20"] = qmetrics.q20_bases
            except:
                extraInfo["q20"] = "-"

            # if we have a paired end run, lets get that report path
            if get_result_type(result,pstore) == 'paired':
                extraInfo["paired"] = paired_end_stats(result.get_report_path())
            else:
                extraInfo["paired"] = ""

            # track the FPS set on Proton instruments
            try:
                overclock = exp.log["overclock"]
                oc = int(overclock)
                if oc == 1:
                    extraInfo["fps"] = "15"
                elif oc == 2:
                    extraInfo["fps"] = "30"
                else:
                    extraInfo["fps"] = "-"
            except:
                extraInfo["fps"] = "-"

            # track the status of the covereage plugin
            try:
                # pluginDict = ast.literal_eval(pstore["coverageAnalysis"])
                pluginDict = pstore["coverageAnalysis"]
                if pluginDict is not None:
                    extraInfo["cov"] = pluginDict["Target base coverage at 20x"]
            except:
                extraInfo["cov"] = " "

            # track the duplicate rate
            try:
                # pluginDict = ast.literal_eval(pstore["duplicateReads_useAdapter"])
                pluginDict = pstore["duplicateReads_useAdapter"]
                if pluginDict is not None:
                    val = float(pluginDict["duplicate_rate_chr1"])
                    val = val * 100.0
                    extraInfo["dup"] = str(val) + "%"
            except:
                extraInfo["dup"] = " "

            # track the variant caller SNP & InDel accuracy
            try:
                # pluginDict = ast.literal_eval(pstore["validateVariantCaller"])
                pluginDict = pstore["validateVariantCaller"]
                if pluginDict is not None:
                    SNPAccuracy = round(float(pluginDict["InDel_ConsensusAccuracy-AllPos"]) * 100.0, 4)
                    InDelAccuracy = round(float(pluginDict["SNP_ConsensusAccuracy-AllPos"]) * 100.0, 4)
                    extraInfo["vc"] = str(SNPAccuracy) + "% : " + str(InDelAccuracy) + "%"
            except:
                extraInfo["vc"] = " "

            # track the systematic error
            try:
                # pluginDict = ast.literal_eval(pstore["SystematicErrorAnalysis"])
                pluginDict = pstore["SystematicErrorAnalysis"]
                if pluginDict is not None:
                    # print 'Got syserr plugin info: %s' % pluginDict
                    # print 'object is type: %s' % type(pluginDict)
                    if pluginDict["barcoded"]:
                        # print 'it is barcoded'
                        bc = pluginDict["barcodes"]
                        val = 1.0
                        for name in bc:
                            if '-' not in bc[name]["positions-with-sse"]:
                                bcval = float(bc[name]["positions-with-sse"])
                                if bcval < val and bcval > 0.0:
                                    val = bcval
                    else:
                        # print 'it is not barcoded'
                        val = float(pluginDict["positions-with-sse"])
                    # print 'plugin val: %s' % val
                    if val > 0.0:
                        val = val * 100.0
                        extraInfo["syserr"] = str(val) + "%"
                    else:
                        extraInfo["syserr"] = " "
                    # print 'plugin syserr is: %s' % extraInfo["syserr"]
            except:
                extraInfo["syserr"] = " "

            # track coverage uniforimity & 20x coverage
            try:
                # pluginDict = ast.literal_eval(pstore["SystematicErrorAnalysis"])
                pluginDict = pstore["coverageAnalysisLite"]
                if pluginDict is not None:
                    bestCov20 = 0.0
                    bestUnif = 0.0
                    if pluginDict["barcoded"]:
                        # print 'it is barcoded'
                        bc = pluginDict["barcodes"]
                        for name in bc:
                            cov20txt = bc[name]["Target base coverage at 20x"]
                            cov20 = float(cov20txt.rstrip('%'))
                            uniftxt = bc[name]["Uniformity of base coverage"]
                            unif = float(uniftxt.rstrip('%'))
                            if (unif < 100.0) and (unif > 0.0):
                                if cov20 > bestCov20:
                                    bestCov20 = cov20
                                    bestUnif = unif
                                
                    else:
                        # print 'it is not barcoded'
                        cov20txt = pluginDict["Target base coverage at 20x"]
                        bestCov20 = float(cov20txt.rstrip('%'))
                        uniftxt = pluginDict["Uniformity of base coverage"]
                        bestUnif = float(uniftxt.rstrip('%'))
                    # print 'plugin val: %s' % val
                    if bestCov20 > 0.0:
                        extraInfo["cov20"] = str(bestCov20) + "%"
                        extraInfo["unif"] = str(bestUnif) + "%"
                    else:
                        extraInfo["cov20"] = " "
                        extraInfo["unif"] = " "
                    # print 'plugin syserr is: %s' % extraInfo["syserr"]
            except:
                extraInfo["cov20"] = " "
                extraInfo["unif"] = " "

            info.append(extraInfo)

    #sort by chipType
    if len(resultsAll) > 1:
      try: 
        resultsAll, rType, rNew, info = zip(*sorted(zip(resultsAll,rType,rNew,info), key=lambda r:r[0].experiment.chipType))
      except:
        pass
    
    
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
    #links = [web_root+res.reportLink for res in resultsAll]
    links = [web_root+"/report/"+str(res.pk) for res in resultsAll]
 
    #find the sum of the q20 bases
    hqBaseSum = 0
    for res,n,tp in zip(resultsAll,rNew,rType):
        if n and (not tp=='thumb') and (not tp=='paired') and res.best_lib_metrics:
            if res.best_lib_metrics.align_sample == 0:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.q20_mapped_bases
            if res.best_lib_metrics.align_sample == 1:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.extrapolated_mapped_bases_in_q20_alignments
            if res.best_lib_metrics.align_sample == 2:
                hqBaseSum = hqBaseSum + res.best_lib_metrics.q20_mapped_bases
    
    tmpl = loader.get_template(TEMPLATE_NAME)
    ctx = template.Context({"reportsOldThumb":
                                [(r,t,lb,l,i) for r,t,lb,l,i,tp,n in zip(resultsAll,tfms,lbms,links,info,rType,rNew) if tp=='thumb' and not n],
                            "reportsOldWhole":
                                [(r,t,lb,l,i) for r,t,lb,l,i,tp,n in zip(resultsAll,tfms,lbms,links,info,rType,rNew) if tp=='full' and not n],
                            "reportsNew":
                                [(r,t,lb,l,i) for r,t,lb,l,i,tp,n in zip(resultsAll,tfms,lbms,links,info,rType,rNew) if tp=='full' and n],
                            "reportsThumbsNew":
                                [(r,t,lb,l,i) for r,t,lb,l,i,tp,n in zip(resultsAll,tfms,lbms,links,info,rType,rNew) if tp=='thumb' and n],
                            "reportsAmplNew":
                                [(r,t,lb,l,i) for r,t,lb,l,i,tp,n in zip(resultsAll,tfms,lbms,links,info,rType,rNew) if tp=='ampl' and n],
                            "reportsPairNew":
                                [(r,t,lb,l,i) for r,t,lb,l,i,tp,n in zip(resultsAll,tfms,lbms,links,info,rType,rNew) if tp=='paired' and n],
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
    if RECIPS:
        send_nightly()

if __name__ == '__main__':
    main(sys.argv)

