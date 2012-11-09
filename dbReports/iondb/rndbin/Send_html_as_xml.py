import sys
import os
import datetime
from os import path
import urllib, urllib2

verbose = 0

def installAuth():
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, "http://updates.iontorrent.com/metrics", 'metrics', 'ionmetrics')
    handler = urllib2.HTTPBasicAuthHandler(password_mgr)
    opener = urllib2.build_opener(handler)
    urllib2.install_opener(opener)

def sendXML(xml_string):
    installAuth()
    query_args = {'xml':xml_string}
    encoded_args = urllib.urlencode(query_args)
    url = 'http://updates.iontorrent.com/metrics/recv_html_as_xml.php'
    response = urllib2.urlopen(url, encoded_args).read()
    if verbose > 0:
        print 'Sent xml file, response:\n%s\n' % response
    if verbose > 1:
        print 'XML File:\n%s\n' % xml_string

if __name__== "__main__":
    siteName = 'Baylor'
    argc = len(sys.argv)
    argcc = 1
    htmlfile = 'default.html'
    while argcc < argc:
        if sys.argv[argcc] == '--htmlfile':
            argcc = argcc + 1
            htmlfile = sys.argv[argcc]
        if sys.argv[argcc] == '--verbose':
            verbose = verbose + 1

        argcc = argcc + 1

    if verbose > 0:
        print 'Opening up html file: %s' % htmlfile
    htmlFile = open(htmlfile, 'r')
    htmlStream = htmlFile.read()

    xmlStream = '<?xml version="1.0"?>\n'
    xmlStream = xmlStream + '<Metrics>\n'
    xmlStream = xmlStream + '    <Site>\n'
    xmlStream = xmlStream + '        <Name>' + siteName + '</Name>\n'
    xmlStream = xmlStream + '        <HTMLName>' + htmlfile + '</HTMLName>\n'
    xmlStream = xmlStream + '        <HTMLData><![CDATA[' + htmlStream + ']]></HTMLData>\n'
    xmlStream = xmlStream + '    </Site>\n'
    xmlStream = xmlStream + '</Metrics>\n'

    sendXML(xmlStream)

