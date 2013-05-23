#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import datetime, os, sys, traceback
import simplejson as json
from operator import itemgetter
import pytz
import subprocess

from iondb.bin.djangoinit import *
from iondb.rundb.models import PluginResult, LibMetrics
# ignore django 'naive datetime' warning
import warnings
warnings.filterwarnings("ignore", module='django.db.models.fields')

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import Encoders

#mailserver = '10.10.20.49'
mailserver = 'smtp.ite'
smtp_port = 25
auth = ("ionadmin","ionadmin") # Authentication for API queries

def get_recips():
  ret = ["alla.shundrovsky@lifetech.com"]
  #ret = ["tony.xu@lifetech.com"]
  #ret = ["c-Daniel.Cuevas2@lifetech.com"]
  #ret = ["c-Daniel.Cuevas2@lifetech.com","!ION-Amplicon@lifetech.com"]
  return ret

RECIPS = get_recips()
SENDER = "donotreply@iontorrent.com"

SiteList = {
    'Proton_East':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.ite'
    },
    'Proton_West':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.itw'
    },
    'Proton_Carlsbad':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.cbd'
    },
    'Proton_Beverly':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'blackbird.bev'
    },
    'ioneast':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'ioneast.ite'
    },
    'ionwest':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'ionwest.itw'
    },
    'beverly':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'aruba.bev'
    },
    'socal':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'carlsbad.cbd'
    },
    'local':{
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': 'localhost'
    },
}
for site in SiteList:
    settings.DATABASES[site] = SiteList[site]

################

def initdoc(f, header):
    f.write('\\documentclass[letterpaper,10pt]{article}\n')
    f.write('\\usepackage{booktabs}\n')
    f.write('\\usepackage{colortbl}\n')
    f.write('\\usepackage{amsmath}\n')
    f.write('\\usepackage{xcolor}\n')
    f.write('\\usepackage{graphicx}\n')
    f.write('\\usepackage[colorlinks=true, urlcolor=blue]{hyperref}\n')
    f.write('\\usepackage{fancyhdr}\n')
    f.write('\\usepackage{array}\n')
    f.write('\\usepackage[T1]{fontenc}\n')

    f.write('\\colorlet{tableheadcolor}{gray!25}\n')
    f.write('\\colorlet{tablerowcolor}{gray!10}\n')
    f.write('\\newcommand{\\headcol}{\\rowcolor{tableheadcolor}}\n')
    f.write('\\newcommand{\\rowcol}{\\rowcolor{tablerowcolor}}\n')
    f.write('\\newcommand{\\topline}{\\arrayrulecolor{black}\\specialrule{0.1em}{\\abovetopsep}{0pt}\\arrayrulecolor{tableheadcolor}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\midline}{\\arrayrulecolor{tableheadcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\lightrulewidth}{0pt}{0pt}\\arrayrulecolor{white}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinecw}{\\arrayrulecolor{tablerowcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\lightrulewidth}{0pt}{0pt}\\arrayrulecolor{white}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinewc}{\\arrayrulecolor{white}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\lightrulewidth}{0pt}{0pt}\\arrayrulecolor{tablerowcolor}\\specialrule{\\belowrulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinew}{\\arrayrulecolor{white}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\rowmidlinec}{\\arrayrulecolor{tablerowcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}}\n')
    f.write('\\newcommand{\\bottomline}{\\arrayrulecolor{white}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\heavyrulewidth}{0pt}{\\belowbottomsep}}\n')
    f.write('\\newcommand{\\bottomlinec}{\\arrayrulecolor{tablerowcolor}\\specialrule{\\aboverulesep}{0pt}{0pt}\\arrayrulecolor{black}\\specialrule{\\heavyrulewidth}{0pt}{\\belowbottomsep}}\n')
    f.write('\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')
    f.write('\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')
    f.write('\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')
    f.write('\\newcolumntype{H}[1]{>{\\color{red}\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}\n')
    #f.write('\\renewcommand{\\arraystretch}{0.8}\n')
    f.write('\\renewcommand{\\tabcolsep}{0pt}\n')

    f.write(''
        '\\makeatletter\n'
        '\\def\\maxwidth{%\n'
        '\\ifdim\\Gin@nat@width>\\linewidth\n'
        '\\linewidth\n'
        '\\else\n'
        '\\Gin@nat@width\n'
        '\\fi\n'
        '}\n'
        '\\makeatother\n')

    # f.write('\\textwidth 7.5in\n')
    # f.write('\\marginsize{0.5in}{0.5in}{1.0in}{1.0in}\n')
    f.write('\\usepackage[top=0.85in, bottom=0.85in, left=0.5in, right=0.5in]{geometry}\n')

    f.write(''
        '\\fancypagestyle{mystyle}{%\n'
        '\\fancyhead{}                                                    % clears all header fields.\n'
        '\\fancyhead[C]{\\large' +header+ '}                              % title of the report, centered\n'
        '\\fancyfoot{}                                                    % clear all footer fields\n'
        '\\fancyfoot[L]{\\thepage}\n'
        '\\fancyfoot[R]{\\includegraphics[width=20mm]{/opt/ion/iondb/media/IonLogo.png}}\n'
        '\\renewcommand{\\headrulewidth}{1pt}                             % the header rule\n'
        '\\renewcommand{\\footrulewidth}{0pt}\n'
        '}\n')

    f.write('\\begin{document}\n')
    f.write('\\pagestyle{mystyle}\n')

def table_begin(f,header,formatText):
    f.write(''
        '\\begin{tabular}{%s}\n'
        '\\topline\n'
        '\\headcol %s \\\\\n'
        '\\midline\n' % (formatText, header))

def latex(text):
    text = text.replace('_', '\\_').replace('%','\\%')
    return text

################

def generate_content(binList, binOrder, resultsList, sortedList, scoreList, sortedScoreList, aq20List, sortedAQ20List):
    # generate the LaTeX content
    print "Generating content."
    reportFileName = 'AmpliconWeeklyReport' + str(datetime.date.today()) + '.tex'
    header = ' Amplicon Report Summary for ' + datetime.datetime.now().strftime("%A %B %d, %Y")
    f = open(reportFileName, 'w')
    initdoc(f, header)
    f.write('\\begin{tabular}{l}\n'
            '\\small $^\\ast$Note: Score is calculated as \\\\ [-0.5ex]\n'
            '\\footnotesize (0.5$\\times$ "Target Coverage at 20X - Norm 100") $+$ (0.25$\\times$ "Percent Bases with No Strand Bias") $+$ (0.25$\\times$ "Percent All Reads On Target")\\\\\n'
            '\\small $^\\ast$Newly added top runs are highlighted.\n'
            '\\end{tabular}\n'
            )
    
    # Top 5 Binned Runs
    f.write('\\section*{\\normalsize Top 5 Runs based on Target Coverage at 20X - Norm 100 }\n')
    for bin in binOrder:
        if len(resultsList[bin].keys()) == 0: continue
        if dev: print "Top 5 Binned Runs, bin=", bin
        
        f.write('\\subsection*{\\normalsize ' +binList[bin]+ '}\n')

        formatText = '>{\\footnotesize}L{5.5cm}' + '>{\\footnotesize}L{1cm}' + '>{\\footnotesize}L{0.8cm}' \
            + '>{\\footnotesize}C{1.4cm}' + '>{\\footnotesize}H{1.5cm}' + '>{\\footnotesize}C{1.5cm}'*6
        
        header = 'Run Name & Site & Date & Number of Targets & Target Coverage at 20X - Norm 100 & Per Base Accuracy & Percent All Reads On Target & Percent Bases with No Strand Bias & Score & Combined TPR at 20X & Combined PPV at 20X'
        f.write('\\begin{tabular}{%s}\n'
                '\\topline\n'
                '\\headcol %s \\\\\n'
                '\\midline\n' % (formatText, header))
        
        for count, currRun in enumerate(sortedList[bin]):
            runDict = resultsList[bin][currRun]
            if dev:  print runDict
            
            rowText = '%d.%s & %s & %s & %s & %s & %s & %s & %s & %.4f & %s & %s' % (
                count+1,
                latex('\\href{' +runDict["sitelink"]+ '}{' +currRun+ '}'),
                runDict["site"],
                runDict["timeStamp"].strftime("%Y-%m-%d"),
                str(runDict["numTargets"]),
                latex(runDict["target_coverage_at_20x_-_norm_100"]),
                latex(runDict["per_base_accuracy"]),
                latex(runDict["percent_all_reads_on_target"]),
                latex(runDict["percent_no_strand_bias_of_all_bases"]),
                runDict["score"],
                latex(str(runDict["vvcCombSAP"])),
                latex(str(runDict["vvcCombPPV"])),
            )
            if runDict["timeStamp"] >= timeRangeNew:
                f.write('\\rowcolor{yellow!50}\n')
            f.write(''+rowText+' \\\\\n')
        
        f.write('\\bottomlinec\n' '\\end{tabular}\n')

    # Top 10 AQ20 Runs
    if dev: print "Top 10 AQ20 Runs"
    f.write('\\section*{\\normalsize Top 10 Runs Based on AQ20 Bases }\n')
    
    formatText = '>{\\footnotesize}L{6cm}' + '>{\\footnotesize}L{1cm}' + '>{\\footnotesize}L{0.8cm}' \
            + '>{\\footnotesize}C{1.9cm}' + '>{\\footnotesize}H{1.8cm}' + '>{\\footnotesize}C{1.9cm}'*4

    header = 'Run Name & Site & Date & Total Number of Reads & AQ20 Bases (Mbp) & AQ20 Alignments & Homopolymer Accuracy (6mer) & Combined TPR at 20X & Combined PPV at 20X'
    f.write('\\begin{tabular}{%s}\n'
            '\\topline\n'
            '\\headcol %s \\\\\n'
            '\\midline\n' % (formatText, header))
    
    for count, currRun in enumerate(sortedAQ20List):
        runDict = aq20List[currRun]
        if dev:  print runDict
        
        rowText = '%d.%s & %s & %s & %s & %.2f &l %s & %s & %s & %s' % (
            count+1,
            latex('\\href{' +runDict["sitelink"]+ '}{' +currRun+ '}'),
            runDict["site"],
            runDict["timeStamp"].strftime("%Y-%m-%d"),
            str(runDict["numReads"]),
            float(float(runDict["aq20bp"])/1000000.0),
            str(runDict["aq20align"]),
            latex(runDict["hp6mer"]),
            latex(str(runDict["vvcCombSAP"])),
            latex(str(runDict["vvcCombPPV"])),
        )
        if runDict["timeStamp"] >= timeRangeNew:
            f.write('\\rowcolor{yellow!50}\n')
        f.write(''+rowText+' \\\\\n')
    
    f.write('\\bottomlinec\n' '\\end{tabular}\n')

    # Top 20 Score Table
    if dev: print "Top 20 Score Runs"
    f.write('\\section*{\\normalsize Top 20 Runs Based on Score }\n')
    
    formatText = '>{\\footnotesize}L{5.5cm}' + '>{\\footnotesize}L{1cm}' + '>{\\footnotesize}L{0.8cm}'\
        + '>{\\footnotesize}C{1.6cm}'*5 + '>{\\footnotesize}H{1cm}' + '>{\\footnotesize}C{1.5cm}'*2
    
    header = 'Run Name & Site & Date & Number of Targets & Target Coverage at 20X - Norm 100 & Per Base Accuracy & Percent All Reads On Target & Percent Bases with No Strand Bias & Score & Combined TPR at 20X & Combined PPV at 20X'
    f.write('\\begin{tabular}{%s}\n'
            '\\topline\n'
            '\\headcol %s \\\\\n'
            '\\midline\n' % (formatText, header))
    
    for count, currRun in enumerate(sortedScoreList):
        runDict = scoreList[currRun]
        if dev:  print runDict
        
        rowText = '%d.%s & %s & %s & %s & %s & %s & %s & %s & %.4f & %s & %s' % (
            count+1,
            latex('\\href{' +runDict["sitelink"]+ '}{' +currRun+ '}'),
            runDict["site"],
            runDict["timeStamp"].strftime("%Y-%m-%d"),
            str(runDict["numTargets"]),
            latex(runDict["target_coverage_at_20x_-_norm_100"]),
            latex(runDict["per_base_accuracy"]),
            latex(runDict["percent_all_reads_on_target"]),
            latex(runDict["percent_no_strand_bias_of_all_bases"]),
            runDict["score"],
            latex(str(runDict["vvcCombSAP"])),
            latex(str(runDict["vvcCombPPV"])),
        )
        if runDict["timeStamp"] >= timeRangeNew:
            f.write('\\rowcolor{yellow!50}\n')
        f.write(''+rowText+' \\\\\n')
    
    f.write('\\bottomlinec\n' '\\end{tabular}\n')

    # Finally
    f.write('\\end{document}\n')
    f.close()
    return reportFileName

def sendMail(sender,recips,subject,attachment):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ", ".join(recips)
    
    msg.attach( MIMEText('Amplicon Summary Report is attached.') )
    try:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(attachment,"rb").read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(attachment))
        msg.attach(part)
        
        s = smtplib.SMTP(mailserver, smtp_port)
        s.sendmail(sender, recips, msg.as_string())
        s.quit()
        print "Email sent sucessfully."
    except:
        print >> sys.stderr, "Error: Could not send email."
        print traceback.print_exc()

def calcScore(data):
    # Calculate score using variety of amplicon metrics
    try:
        # Be sure that these value actually exist
        # Return 0 if not
        tarCov = float(data["target_coverage_at_20x_-_norm_100"].rstrip("%"))
        perStrBias = float(data["percent_no_strand_bias_of_all_bases"].rstrip("%"))
        perAllReads = float(data["percent_all_reads_on_target"].rstrip("%"))
        score = 0.5*tarCov + 0.25*perStrBias + 0.25*perAllReads
        return score
    except:
        return 0
        

def isAmplicon(runDict):
    # Determine if the run is an amplicon run by checking if AmpliconStats or ampliconGeneralAnalysis was successful
    # Return false of neither was successful
    # Return which plugin completed (priority goes to AmpliconStats)
    #print "running into isAmplicon for %s" % runDict["resultsName"]

    if (not runDict["pluginStore"].has_key("AmpliconStats") or
        (runDict["pluginStore"]["AmpliconStats"].has_key("barcodes") and len(runDict["pluginStore"]["AmpliconStats"]["barcodes"]) == 0) or
        (not runDict["pluginStore"]["AmpliconStats"].has_key("barcodes") and len(runDict["pluginStore"]["AmpliconStats"]) == 0)):
        
        if (not runDict["pluginStore"].has_key("ampliconGeneralAnalysis") or
            len(runDict["pluginStore"]["ampliconGeneralAnalysis"]) == 0):
            #print "return false for isAmplicon from condition 1"
            return False
        else:
            #print "return ampGenAnalysis"
            return "ampliconGeneralAnalysis"

    #NOTE: this needs to be more restrictive than simply scanning resultsName, otherwise records can be accidentally removed.
    #       many database fields could be used to filter on (projects, samples etc.)
    # manually added to remove some targetReseq runs. Need a mechanism to differentiate these runs automatically
    if (runDict["resultsName"].lower().find('huref_ex') != -1):
        #print "%s is a huref targetreseq." % runDict["resultsName"];
        return False
    if (runDict["resultsName"].lower().find('318_ecoli') != -1):
        #print "%s is an ecoli run." % runDict["resultsName"];
        return False
    if (runDict["resultsName"].lower().find('165777') != -1):
        #print "%s is an ecoli run." % runDict["resultsName"];
        return False
    
    if (runDict["pluginStore"].has_key("variantCaller") and runDict["pluginStore"]["variantCaller"].has_key('Library Type')):
        #print "library type %s" % runDict["pluginStore"]["variantCaller"]['Library Type'];
        if (runDict["pluginStore"]["variantCaller"]["Library Type"].lower().find('ampliseq') == -1):
            #print "This is not an ampliseq run";
            return False

    return "AmpliconStats"

def getAlignmentInfo(libmetrics):
    # Query TS to obtain AQ numbers from library metrics table
    if libmetrics:
        aq20Bases = libmetrics["q20_mapped_bases"]
        aq20Aligns = libmetrics["q20_alignments"]
        numReads = libmetrics["totalNumReads"]
        return (aq20Bases,aq20Aligns,numReads)
    else:
        return (-1,-1,-1)

def getHPInfo(pluginStore):
    # Parse out homopolymer 6mer data from 1_Torrent_Accuracy plugin in results table
    if (not pluginStore.has_key("1_Torrent_Accuracy") or
        not pluginStore["1_Torrent_Accuracy"].has_key("pooled") or
        not pluginStore["1_Torrent_Accuracy"]["pooled"].has_key("alignFlowSignals") or
        not pluginStore["1_Torrent_Accuracy"]["pooled"]["alignFlowSignals"].has_key("Accuracy") or
        not pluginStore["1_Torrent_Accuracy"]["pooled"]["alignFlowSignals"]["Accuracy"].has_key("6") or
        not pluginStore["1_Torrent_Accuracy"]["pooled"]["alignFlowSignals"]["Accuracy"]["6"].has_key("All")):
        return "N/A"
    return "%.2f" % float(pluginStore["1_Torrent_Accuracy"]["pooled"]["alignFlowSignals"]["Accuracy"]["6"]["All"])

def getAmpliconData(runDict):
    # For each metric, set to 0% if unable to access (mostly found in older runs)
    tarCov20xNorm = runDict.get("target_coverage_at_20x_-_norm_100", "0%")
    perBaseAcc = runDict.get("per_base_accuracy", "0%")
    perAllReadsOnTarg = runDict.get("percent_all_reads_on_target", "0%")
    perNoStrandBiasBases = runDict.get("percent_no_strand_bias_of_all_bases", "0%")
    numTargets = int(runDict.get("number_of_targets", 0))
    coverage20x = runDict.get("target_coverage_at_20x", "NA")
    score = calcScore(runDict)
    return (tarCov20xNorm,perBaseAcc,perAllReadsOnTarg,perNoStrandBiasBases,numTargets,coverage20x,score)

def getVVCInfo(pluginStore, bc=''):
    #string "line" may have IonXpress_xxx in it, need this to determine which barcode
    vvcSNPSAP = "NA"
    vvcSNPCAAP = "NA"
    vvcInDelSAP = "NA"
    vvcSNPTP = "NA"
    vvcSNPFP = "NA"
    vvcInDelTP = "NA"
    vvcInDelFP = "NA"
    vvcSNPPPV = "NA"
    vvcInDelPPV = "NA"
    vvcSNPFN = "NA"
    vvcInDelFN = "NA"
    vvcCombSAP = "NA" #combined sensitivity
    vvcCombPPV = "NA" #combined sensitivity
    if (pluginStore.has_key("validateVariantCaller")):
        vvcDict = pluginStore["validateVariantCaller"]
        if (vvcDict.get("barcoded","").lower() == "true" or vvcDict.get("barcoded","").lower() == "t" ) and vvcDict.has_key("barcodes"):
            if pluginStore["validateVariantCaller"]["barcodes"].get(bc):
                (vvcSNPSAP,vvcSNPCAAP,vvcInDelSAP,vvcSNPTP,vvcSNPFP,vvcInDelTP,vvcInDelFP,vvcSNPFN,vvcInDelFN) = getValidateVCData(pluginStore["validateVariantCaller"]["barcodes"][bc])
        else: #non-barcoded
            (vvcSNPSAP,vvcSNPCAAP,vvcInDelSAP,vvcSNPTP,vvcSNPFP,vvcInDelTP,vvcInDelFP,vvcSNPFN,vvcInDelFN) = getValidateVCData(pluginStore["validateVariantCaller"])

    if (vvcSNPTP != "NA" and vvcSNPFP != "NA"):
        totalP = float(vvcSNPTP) + float(vvcSNPFP);
        thetp = float(vvcSNPTP) * 100;
        try:
            vvcSNPPPV = (thetp / totalP)
            vvcSNPPPV = str(round(vvcSNPPPV, 2))+"%"
        except:
            vvcSNPPPV = -1
    if (vvcInDelTP != "NA" and vvcInDelFP != "NA"):
        totalP = float(vvcInDelTP) + float(vvcInDelFP);
        thetp = float(vvcInDelTP) * 100;
        try:
            vvcInDelPPV = (thetp / totalP)
            vvcInDelPPV = str(round(vvcInDelPPV, 2))+"%"
        except:
            vvcInDelPPV = -1

    #combined sensitivity and PPV
    if (vvcSNPTP != "NA" and vvcSNPFN != "NA" and vvcInDelTP != "NA" and vvcInDelFN != "NA"):
        totalP = float(vvcSNPTP) + float(vvcSNPFN) + float(vvcInDelTP) + float(vvcInDelFN);
        thetp = (float(vvcSNPTP) + float(vvcInDelTP))* 100;
        try:
            vvcCombSAP = (thetp / totalP)
            vvcCombSAP = str(round(vvcCombSAP, 2))+"%"
        except:
            vvcCombSAP = -1
    if (vvcSNPTP != "NA" and vvcSNPFP != "NA" and vvcInDelTP != "NA" and vvcInDelFP != "NA"):
        totalP = float(vvcSNPTP) + float(vvcSNPFP) + float(vvcInDelTP) + float(vvcInDelFP);
        thetp = (float(vvcSNPTP) + float(vvcInDelTP))* 100;
        try:
            vvcCombPPV = (thetp / totalP)
            vvcCombPPV = str(round(vvcCombPPV, 2))+"%"
        except:
            vvcCombPPV = -1

    if vvcSNPSAP != "NA":
        if vvcSNPSAP != -1:
            vvcSNPSAP = round(float(vvcSNPSAP), 2)
            vvcSNPSAP = str(vvcSNPSAP)+"%"
    if vvcSNPCAAP != "NA":
        if vvcSNPCAAP != -1:
            vvcSNPCAAP = round(float(vvcSNPCAAP)*100, 2)
            vvcSNPCAAP = str(vvcSNPCAAP)+"%"
    if vvcInDelSAP != "NA":
        if vvcInDelSAP != -1:
            vvcInDelSAP = round(float(vvcInDelSAP), 2)
            vvcInDelSAP = str(vvcInDelSAP)+"%"

    ret = {
        'vvcSNPSAP':vvcSNPSAP,
        'vvcSNPCAAP':vvcSNPCAAP,
        'vvcInDelSAP':vvcInDelSAP,
        'vvcSNPPPV':vvcSNPPPV,
        'vvcInDelPPV':vvcInDelPPV,
        'vvcCombSAP':vvcCombSAP,
        'vvcCombPPV':vvcCombPPV
    }
    return (ret)

def getValidateVCData(runDict):
    # For each metric, set to 0% if unable to access (mostly found in older runs)
    SNP_SensitivityAllPos = runDict.get("SNP_Sensitivity-AllPos",-1)
    SNP_ConsensusAccuracyAllPos = runDict.get("SNP_ConsensusAccuracy-AllPos",-1)
    SNP_SensitivityGE20x = runDict.get("SNP_Sensitivity>=20x",-1)
    SNP_ConsensusAccuracyGE50x = runDict.get("SNP_ConsensusAccuracy>=50x",-1)
    InDel_SensitivityAllPos = runDict.get("InDel_Sensitivity-AllPos",-1)
    InDel_ConsensusAccuracyAllPos = runDict.get("InDel_ConsensusAccuracy-AllPos",-1)
    InDel_SensitivityGE20x = runDict.get("InDel_Sensitivity>=20x",-1)
    InDel_ConsensusAccuracyGE50x = runDict.get("InDel_ConsensusAccuracy>=50x",-1)
    SNP_TPAllPos = runDict.get("SNP_TP-AllPos",-1)
    SNP_TPGE20x = runDict.get("SNP_TP>=20x",-1)
    SNP_FPGE20x = runDict.get("SNP_FP>=20x",-1)
    SNP_FNGE20x = runDict.get("SNP_FN>=20x",-1)
    InDel_TPGE20x = runDict.get("InDel_TP>=20x",-1)
    InDel_FPGE20x = runDict.get("InDel_FP>=20x",-1)
    InDel_FNGE20x = runDict.get("InDel_FN>=20x",-1)
    SNP_FPAllPos = runDict.get("SNP_FP-AllPos",-1)
    InDel_TPAllPos = runDict.get("InDel_TP-AllPos",-1)
    InDel_FPAllPos = runDict.get("InDel_FP-AllPos",-1)

    #return (SNP_SensitivityAllPos,SNP_ConsensusAccuracyAllPos,InDel_SensitivityAllPos,SNP_TPAllPos,SNP_FPAllPos,InDel_TPAllPos,InDel_FPAllPos)
    #requested by Mark, now report sensitivity of >= 20x and its PPV
    return (SNP_SensitivityGE20x,SNP_ConsensusAccuracyAllPos,InDel_SensitivityGE20x,SNP_TPGE20x,SNP_FPGE20x,InDel_TPGE20x,InDel_FPGE20x,SNP_FNGE20x,InDel_FNGE20x)

def getBinName(numTargets):
    bin = ""
    if numTargets < 50: bin = "lt50"
    elif numTargets <= 100: bin = "50to100"
    elif numTargets <= 200: bin = "100to200"
    elif numTargets <= 1000: bin = "200to1000"
    elif numTargets <= 5000: bin = "1000to5000"
    else: bin = "gt5000"
    #print "numTargets is binned to %s" % bin
    #print "numTargets is %d " % numTargets

    return bin

def processNewRuns(records,resultsList,scoreList,aq20List):
    # Process runs and check if each run or barcode can be placed into one of the tables
    print "Processing runs ...."
    
    try:
        count = 0 # Counter for number of amplicon runs    for terminal message purposes
        # Iterate through all runs
        for runDict in records.values():
            # If new runs are queryed, must filter out those that are not "amplicon" related
            # Check if the new ampliconStats plugin was run
            name = isAmplicon(runDict)
            if not name:
                #print "not pluginStore, loop continue"
                continue
            count += 1
            
            # For consistency, set structure to mimic barcode structure if barcodes do not exist
            barcoded = True
            ampArr = runDict["pluginStore"][name]
            if not ampArr.has_key("barcoded") or ampArr["barcoded"] == "false":
                ampArr["barcodes"] = {"-1":ampArr}
                barcoded = False

            id = runDict['pk']
            timeStamp = runDict['timeStamp']
            expId = runDict['experiment_id']            
            site = runDict['site']
            link = runDict['link']
            path = ampArr['pluginPath']

            # Iterate through amplicon plugin data for each barcode
            # Will be > 1 if run contained barcodes
            for bc,bcArr in ampArr["barcodes"].items():
                runName = runDict["resultsName"]+"_"+str(id)
                #print "run name %s" % runName
                # Append barcode if applicable
                if barcoded: runName += "-"+bc
            
                # Grab data fields and run info
                (tarCov20xNorm,perBaseAcc,perAllReadsOnTarg,perNoStrandBiasBases,numTargets,coverage20x,score) = getAmpliconData(bcArr)
                numAQ20Bases,numAQ20Aligns,numReads = getAlignmentInfo(runDict.get("LibMetrics"))
                hp6mer = getHPInfo(runDict["pluginStore"])
                VVCInfo = getVVCInfo(runDict["pluginStore"], bc)
                
                # Determine bin hash key (used for target coverage tables)
                bin = getBinName(numTargets)
            
                # Determine file path to AmpliconStats web page
                ampliconLink = ""
                if name == "ampliconGeneralAnalysis":
                    ampliconLink = path +"AmpliconGeneralAnalysis.html"
                elif barcoded:
                    ampliconLink = path +bc+"/AmpliconStats.html"
                else:
                    ampliconLink = path +"AmpliconStats.html"
                    
                # Add this record to resultsList,scoreList,aq20List used to generate output tables
                resultsList[bin][runName] = {
                        "sitelink":ampliconLink,
                        "site":site, "id":id, "expId":expId,
                        "numTargets":numTargets,
                        "target_coverage_at_20x_-_norm_100":tarCov20xNorm,
                        "per_base_accuracy":perBaseAcc,
                        "percent_all_reads_on_target":perAllReadsOnTarg,
                        "percent_no_strand_bias_of_all_bases":perNoStrandBiasBases,
                        "coverage20x":coverage20x,
                        "timeStamp":timeStamp,
                        "score":score,
                        "barcode":bc
                }
                resultsList[bin][runName].update(VVCInfo)
    
                scoreList[runName] = {
                        "sitelink":ampliconLink,
                        "site":site, "id":id, "expId":expId,
                        "numTargets":numTargets,
                        "target_coverage_at_20x_-_norm_100":tarCov20xNorm,
                        "per_base_accuracy":perBaseAcc,
                        "percent_all_reads_on_target":perAllReadsOnTarg,
                        "percent_no_strand_bias_of_all_bases":perNoStrandBiasBases,
                        "coverage20x":coverage20x,
                        "timeStamp":timeStamp,
                        "score":score, 
                        "barcode":bc
                }
                scoreList[runName].update(VVCInfo)
                
                aq20List[runName] = {
                        "sitelink":link,
                        "site":site, "id":id, "expId":expId,
                        "aq20bp":numAQ20Bases,
                        "aq20align":numAQ20Aligns,
                        "numReads":numReads,
                        "hp6mer":hp6mer,
                        "timeStamp":timeStamp
                }
                aq20List[runName].update(VVCInfo)
                
        print "   %s amplicon records found.\n" % count
            
    except Exception:
        print >> sys.stderr,"FATAL ERROR"
        print >> sys.stderr,traceback.print_exc()
        try:
            print >> sys.stderr,"resultsName=",runDict["resultsName"]
        except Exception:
            print >> sys.stderr,"No results name found yet"
            print >> sys.stderr,traceback.print_exc()
            try:
                print >> sys.stderr,"runDict=",runDict
            except Exception:
                print >> sys.stderr,"No runDict either"
                print >> sys.stderr,"====dbOutput====\n%s" % dbOutput
                print >> sys.stderr,traceback.print_exc()
        sys.exit(1)

def sort_runs(sortDict, key, num=20, tie_key=''):
    # Sorts runs by given key, returns sorted list with names for 20 records
    # for reanalyzed results returns the highest score record
    ret = []
    to_sort = []
    for name, runDict in sortDict.iteritems():
        currInfo = [name]
        if key == "target_coverage_at_20x_-_norm_100":
            currInfo.append(int(100*float(runDict[key].rstrip("%"))))
        else:
            currInfo.append(runDict[key])
        if tie_key:
            currInfo.append(runDict[tie_key])
        to_sort.append(currInfo)
    
    if tie_key:
        sortedList = sorted(to_sort,key=itemgetter(1,2),reverse=True)
    else:
        sortedList = sorted(to_sort,key=itemgetter(1),reverse=True)
    
    exp_unique = []
    for l in sortedList:
        name = l[0]
        unique = sortDict[name]['site'] + str(sortDict[name]['expId'])
        if  unique not in exp_unique:
            exp_unique.append(unique)
            ret.append(name)
        if len(ret) == num:
            break
        
    if dev:
        print 'Sort results by %s %s' % (key, tie_key)
        for i,s in enumerate(ret):
            if tie_key:
                print '(%d) %s\t %s\t %s' % (i,s,sortDict[s][key],sortDict[s][tie_key])
            else:
                print '(%d) %s\t %s' % (i,s,sortDict[s][key])
        print '\n'
    return ret
    
def run(records):
    binList = {"lt50":"Amplicons < 50", "50to100":"50 < Amplicons < 100",
                "100to200":"100 < Amplicons < 200","200to1000":"200 < Amplicons < 1000",
                "1000to5000":"1000 < Amplicons < 5000","gt5000":"Amplicons > 5000"} # Mapping from bin to display bin name
    binOrder = ["lt50","50to100","100to200","200to1000","1000to5000","gt5000"] # Order to print out bin in email
    resultsList = {"lt50":{},"50to100":{},"100to200":{},"200to1000":{},"1000to5000":{},"gt5000":{}} 
    sortedList = {"lt50":[],"50to100":[],"100to200":[],"200to1000":[],"1000to5000":[],"gt5000":[]} # Holds sorted top 5
    scoreList = {} # Holds top score
    sortedScoreList = [] # Holds sorted top score
    aq20List = {} # Holds top 10 aq20
    sortedAQ20List = [] # Holds sorted top aq20

    processNewRuns(records, resultsList=resultsList,scoreList=scoreList,aq20List=aq20List)
    #print "\n resultsList: \n", json.dumps(resultsList, indent=1)
    #print "\n scoreList: \n", json.dumps(scoreList, indent=1)
    #print "\n aq20List: \n", json.dumps(aq20List, indent=1)
    
    # Sort and store top results in sorted lists
    print "Sorting binned list."
    for bin in binOrder:
        if len(resultsList[bin].keys()) > 0:
            sortedList[bin] = sort_runs(resultsList[bin], "target_coverage_at_20x_-_norm_100", tie_key="score", num=5)
    print json.dumps(sortedList, indent=2)
    
    print "Sorting score list."
    sortedScoreList = sort_runs(scoreList, "score", num=20)
    print json.dumps(sortedScoreList, indent=2)

    print "Sorting aq20 list."
    sortedAQ20List = sort_runs(aq20List, "aq20bp", num=10)
    print json.dumps(sortedAQ20List, indent=2)
    
    # generate report content (writes .tex file)
    reportFileName = generate_content(binList, binOrder, resultsList, sortedList, scoreList, sortedScoreList, aq20List, sortedAQ20List)

    # generate pdf file
    cmd = ['pdflatex', reportFileName]
    print 'Running "%s"' % ' '.join(cmd)
    with open(os.devnull, "w") as devnull:
        subprocess.call(cmd, stdout=devnull)

    #Email
    subject = "Amplicon Weekly Summary"
    pdfName = reportFileName.replace('.tex', '.pdf')
    sendMail(SENDER,RECIPS,subject,pdfName)
    
    
################
    
if __name__== "__main__":
    dev = False
    if len(sys.argv) > 1 and sys.argv[1] == "dev": dev = True
    if dev: print "In dev mode!"
    
    sites = ['ioneast', 'ionwest', 'socal']
    #sites = ['ionwest']
    print 'AmpliconWeeklyReport started.\nSites:', ', '.join([SiteList[site]['HOST'] for site in sites])
    
    # Results analyzed within this number of days will be highlighted as new.
    numDaysNew = 7
    timeRangeNew = datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=(numDaysNew))

    # The following data will be retrieved:
    amplicon_plugins = ['AmpliconStats','ampliconGeneralAnalysis']
    extra_plugins = ['1_Torrent_Accuracy','variantCaller','validateVariantCaller']
    results_fields = ('pk', 'resultsName', 'timeStamp', 'experiment_id')
    libmetrics_fields = ("results__pk", "q20_mapped_bases", "q20_alignments", "totalNumReads")

    ''' Here we will gather all needed data in minimal number of queries to optimize DB access '''
    records = {}
    for site in sites:
        host = SiteList[site]['HOST']
        print 'Getting data from %s ....' % host
        
        pluginresults = PluginResult.objects.using(site).select_related().filter(state='Completed', plugin__name__in= amplicon_plugins)
    
        '''
        print 'TESTING'
        timeRange = datetime.datetime.now() - datetime.timedelta(days=(10))
        pluginresults = pluginresults.filter(result__timeStamp__gte=timeRange)
        '''
        
        result_pks = []
        for pr in pluginresults:
            if len(pr.store)==0:
                continue
            record_key = site + '_' + str(pr.result.pk)
            result_pks.append(pr.result.pk)
            if record_key not in records:
                records[record_key] = {}
            
            for key in results_fields:
                records[record_key][key] = getattr(pr.result, key)
            records[record_key]['site'] = site
            records[record_key]['link'] = 'http://' + host + pr.result.reportLink
            records[record_key].setdefault("pluginStore",{})[pr.plugin.name] = pr.store
            pluginPath = 'http://'+host+pr.result.reportLink+'plugin_out/'+pr.plugin.name+'_out/'
            records[record_key]["pluginStore"][pr.plugin.name]['pluginPath'] = pluginPath
    
        libmetrics = LibMetrics.objects.using(site).filter(results__pk__in=result_pks).values(*libmetrics_fields)
        for libmetric in libmetrics:
            record_key = site + '_' + str(libmetric['results__pk'])
            records[record_key]['LibMetrics'] = libmetric
        #print json.dumps(records[records.keys()[0]], indent=2)
        extra = PluginResult.objects.using(site).select_related().filter(result__pk__in=result_pks, state='Completed', plugin__name__in=extra_plugins)
        for pr in extra:
            record_key = site + '_' + str(pr.result.pk)
            if record_key in records and len(pr.store)>0:
                records[record_key]["pluginStore"][pr.plugin.name] = pr.store

        print '   retrieved %d records.' % len(records)
    
    # generate the report
    run(records)
