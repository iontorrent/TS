# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import division

from pylab import *

import psycopg2
import pandas
import json
import datetime
import numpy as np

import os.path
import urllib
import urllib2
from cookielib import CookieJar

def flatten_dict(dd, separator='_', prefix=''):
    ret = {}
    if isinstance(dd, dict):
        for kk,vv in dd.items():
            for k,v in flatten_dict(vv,separator,kk).items():
                key = prefix + separator + k if prefix else k
                ret[key]=v
        return ret
    else:
        return { prefix : dd }
    
class IonDBData( object ):
    def __init__(self, db):
        self.__db = db
        return

    def uniquify_columns(self, seq ):
        seen = {}
        result = []
        for item in seq:
            if item in seen:
                seen[item] += 1
                item += "_"+str(seen[item])
            else:
                seen[item] = 1
            result.append(item)
        return result

    def df_from_db(self, query ):
        self.__db.query(query)
        rows = self.__db.rows
        if not isinstance(rows, list):
            result = list(rows)
        columns = [col_desc[0] for col_desc in self.__db.cursor.description]

        columns = self.uniquify_columns( columns )
        result = pandas.DataFrame.from_records(rows, columns=columns,
                                    coerce_float=True)
        return result


    def getExplogData( self, data, username, password ):
        host = self.__db._IonDB__host
        data['ChipEfuse']=''
        page = "http://%s/"%(host)
        cookieJar = CookieJar()
        auth = {'username': username, 'password': password, 'remember_me': 'on'}
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookieJar))
        request = urllib2.Request("http://%s/login/" % host, data=urllib.urlencode(auth))
        response = opener.open(request)
        for id,d in data.iterrows():
            url = d['reportLink'].replace('log.html','')
            path = "%s%s/explog.txt"%(page,url)
            try:
                request = urllib2.Request(path)
                explog = opener.open(request)
                for s in explog:
                    if s.find('Chip Efuse:')>-1:
                        s1=s[11:]
                        data=data.set_value(id,'ChipEfuse',s1)
            except Exception,ex:
                print 'Exception:',ex
                pass
        return data

    def getExplogInstrumentData( self, host, username, password, data ):
        data['PGMTemp']=0.
        data['ChipTemp']=0.
        data['PGMPressure']=0.
        data['W1pH']=0.
        data['W2pH']=0.
        data['Noise']=0.
        page = "http://%s/"%(host)
        cookieJar = CookieJar()
        auth = {'username': username, 'password': password, 'remember_me': 'on'}
        opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookieJar))
        request = urllib2.Request("http://%s/login/" % host, data=urllib.urlencode(auth))
        response = opener.open(request)
        for id,d in data.iterrows():
            url = d['reportLink'].replace('log.html','')
            path = "%s%s/explog.txt"%(page,url)
            try:
                request = urllib2.Request(path)
                explog = opener.open(request)
                for s in explog:
                    if s.find('PGMTemperature')>-1:
                        s1=float(s.split()[1])
                        data=data.set_value(id,'PGMTemp',s1)
                    if s.find('ChipTemperature')>-1:
                        s1=float(s.split()[1])
                        data=data.set_value(id,'ChipTemp',s1)
                    if s.find('PGMPressure')>-1:
                        s1=float(s.split()[1])
                        data=data.set_value(id,'PGMPressure',s1)
                    if s.find('W1pH')>-1:
                        s1=float(s.split()[1])
                        data=data.set_value(id,'W1pH',s1)
                    if s.find('W2pH')>-1:
                        s1=float(s.split()[1])
                        data=data.set_value(id,'W2pH',s1)
                    if s.find('Noise')>-1:
                        s1=float(s.split()[1])
                        data=data.set_value(id,'Noise',s1)
            except:
                print "can't open:%s\n"%path
                pass
        return data

    def getPluginData(self,run_df,plugin_list):
        if len(plugin_list)<1:
            return run_df
            
        query="""SELECT rundb_plugin.name,state,store FROM rundb_pluginresult JOIN rundb_plugin ON rundb_plugin.id=rundb_pluginresult.plugin_id WHERE rundb_plugin.name IN %s AND rundb_pluginresult.result_id=%s"""
        plugin_data_list=[]
        for run_id in run_df.index:
            #db.cursor.mogrify(query,(plugin_list,run_id))
            self.__db.cursor.execute(query,(plugin_list,run_id))
            rf={'run_id':run_id}
            for r in self.__db.cursor:
                if r[1]=='Completed':
                    plugin_name=r[0]
                    try:
                        plugin_data=json.loads(r[2])
                        rf.update(flatten_dict(plugin_data,plugin_name+'_'))
                    except Exception as e:
                        #print e
                        #print plugin_data
                        #print r[0], r[2], run_id
                        pass
            plugin_data_list.append(rf)
            self.__db.connection.commit()
            
        plugin_df = pandas.DataFrame(plugin_data_list)
        plugin_df.set_index('run_id',inplace=True)
        data = run_df.join(plugin_df)
        return data

    def __getProjectId(self, run_df):
        query = """SELECT rundb_project.name FROM rundb_project JOIN rundb_results_projects ON  rundb_project.id=rundb_results_projects.project_id WHERE rundb_results_projects.results_id=%s"""
        for run_id in run_df.index:
            try:
                querystr=self.__db.cursor.mogrify(query,(run_id,))
                dat = self.__db.query(querystr)
                project_id = dat[0][0]
                run_df=run_df.set_value(run_id,'project_id',project_id)
            except:
                pass
        return run_df



    def getDataByDate(self,start=(datetime.datetime.now()-datetime.timedelta(days=31)),end=(datetime.datetime.now()-datetime.timedelta(days=0)),plugin_list=()):
        query="""select * from rundb_results JOIN rundb_experiment ON rundb_results.experiment_id=rundb_experiment.id JOIN rundb_libmetrics ON rundb_results.id=rundb_libmetrics.report_id JOIN rundb_analysismetrics ON rundb_results.id=rundb_analysismetrics.report_id where "timeStamp">=%s AND "timeStamp"<=%s ORDER BY "timeStamp" DESC"""
        querystr=self.__db.cursor.mogrify(query,(start,end))
        run_df=self.df_from_db(querystr)
        run_df.set_index('report_id',inplace=True)
        run_df['project_id']=""
        run_df = self.__getProjectId(run_df)
        run_df = self.getPluginData(run_df,plugin_list)
        return run_df

    def getData(self,query,params):
        querystr=self.__db.cursor.mogrify(query,params)
        run_df=self.df_from_db(querystr)
        run_df.set_index('report_id',inplace=True)
        run_df['project_id']=""
        run_df = self.__getProjectId(run_df)
        run_df = self.getPluginData(run_df,plugin_list)
        return run_df

class IonDB(object):
    def __init__(self, host='localhost', database='iondb', user='ion',debug=False):
        super(IonDB, self).__init__()
        self.debug        = debug
        self.__host     = host
        self.__database = database
        self.__user     = user
        conStr = "dbname='%s' user='%s' host='%s'"%(self.__database, self.__user, self.__host)
        self.connection = psycopg2.connect(conStr)
        self.cursor     = self.connection.cursor()
        self.rows         = None
        return

    def query(self, queryString):
        self.cursor.execute(queryString)
        self.rows = self.cursor.fetchall()
        self.connection.commit()
        return self.rows
    
    def __del__(self):
        self.connection.commit()
        self.cursor.close()
        self.connection.close()


