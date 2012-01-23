# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

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


RECIPS = ["!ION-Amplicon@Lifetech.com", "Mel.Davey@Lifetech.com"]
SENDER = "donotreply@iontorrent.com"

def send_html(sender,recips,subject,html):
    msg = mail.EmailMessage(subject,html,sender,recips)
    msg.content_subtype = "html"
    msg.send()

if __name__=="__main__":
    today = datetime.date.today()
    subject = 'Amplicon multi-site report %s' % today
    htmlFile = open("/results/custom_reports/ampl-" + str(datetime.date.today())+ ".html",'r')
    htmlText = htmlFile.read()
    send_html(SENDER, RECIPS, subject, htmlText)
    htmlFile.close()

