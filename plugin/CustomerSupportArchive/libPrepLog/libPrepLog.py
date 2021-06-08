#!/usr/bin/python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import textwrap
import shutil
import json
from subprocess import *
from ion.plugin import *
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime, timedelta
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
from jsonconv import json2html


def plotValues(data, str1, str2, dirstr):
    df1 = data.filter(regex = str1)
    df2 = data.filter(regex = str2)
    df_temp =  pd.concat([df1, df2], axis = 1)
    plt = df_temp.plot( x = str1, rot = 30, title = str2, fontsize = 6).get_figure()
    plt.savefig(os.path.join(dirstr, str2))

def plotVocuum(df, str1, str2, dirstr):
    df1 = df.filter(regex = str1)
    df2 = df.filter(regex = str2)
    df3 = df.filter(regex = 'lane')
    df_temp =  pd.concat([df1, df2, df3], axis = 1)
    df_temp.set_index(str1, inplace = True)
    df = df_temp.groupby('lane')[str2].apply(pd.DataFrame)
    plt = df.plot(legend = True, rot = 30, title = str2, marker='o', fontsize = 6).get_figure()
    plt.savefig(os.path.join(dirstr, str2))

def image_link(imgpath, width = 100):
    ''' Returns code for displaying an image also as a link '''
    text = '<a href="%s"><img src="%s" width="%d%%" /></a>' % ( imgpath, imgpath , width )
    return text


class libPrepLog(IonPlugin):

    version = "1.1.0"
    allow_autorun = True # if true, no additional user input
    runtypes = [ RunType.THUMB, RunType.FULLCHIP, RunType.COMPOSITE ]
    depends = []    
    def launch(self):
        """ main """
        print "running the libPrepLog plugin."
        print "libPrep_log.csv..."
        self.results_dir = os.environ['TSP_FILEPATH_PLUGIN_DIR']
        start_json = getattr(self, 'startpluginjson', None)
        if not start_json:

            try:
                with open(os.path.join(self.results_dir, 'startplugin.json'), 'r') as fh:
                    start_json = json.load(fh)
            except:
                self.log.error("Error reading start plugin json")

        self.results_dir = start_json["runinfo"]["results_dir"] + '/libPrepLog'
        self.raw_data_dir = start_json["runinfo"]["raw_data_dir"]
        self.analysis_dir = start_json["runinfo"]["analysis_dir"] 
        self.plugin_dir = start_json["runinfo"]["plugin_dir"]

        self.runType = start_json["runplugin"]["run_type"]

        if self.runType == 'thumbnail':
            self.raw_data_dir = os.path.dirname(self.raw_data_dir)


        if os.path.exists(os.path.join(self.raw_data_dir, 'libPrep_log.csv')):
            try:
                temp_curr_volt = pd.read_csv(os.path.join(self.raw_data_dir, 'libPrep_log.csv'));  # plots
                self.plotTempCurrentVolt(temp_curr_volt)
            except:
                pass
        else:
            print "libPrep_log.csv does not exist in " +  os.path.join( self.raw_data_dir , 'libPrep_log.csv')

        print "vacuum_log.csv..."
        if os.path.exists(os.path.join(self.raw_data_dir, 'vacuum_log.csv' )):
            try:
                vacuum_log = pd.read_csv(os.path.join(self.raw_data_dir, 'vacuum_log.csv' ), header = None);  # table and plot but parsing needed
                vacuum_head = pd.read_csv(os.path.join(self.raw_data_dir, 'vacuum_log.csv' ));   # read it the second time with header
                vacuum_head = vacuum_head.columns.values
                self.plotVacuumLog(vacuum_log, vacuum_head)
                self.createVacuumLogHTML(vacuum_log)
            except:
                pass
        else:
            print "libPrep_log.csv does not exist in " +  os.path.join( self.raw_data_dir , 'vacuum_log.csv')


        print "ScriptStatus.csv..."
        if os.path.exists(os.path.join(self.raw_data_dir, 'ScriptStatus.csv')):
            script_stats = pd.read_csv(os.path.join(self.raw_data_dir, 'ScriptStatus.csv')); 
            try:
                self.timePerformance(script_stats)
            except:
                pass
        else:
            print "ScriptStatus.csv does not exist in " +  self.raw_data_dir

        print "debug..."
        if os.path.exists(os.path.join(self.raw_data_dir, 'debug')):
            debugLog = os.path.join(self.raw_data_dir, 'debug')
            # get error in debug
            er52lines = self.grepErr(debugLog, 'er52')
            if len(er52lines) > 0:
                df = pd.DataFrame({'Error':er52lines})
                df.to_html(self.results_dir + '/er52.html')
            else:
                df = pd.DataFrame({'Errors in debug':["No er52 Error!"]})
                df.to_html(self.results_dir + '/er52.html')
            # trim debug log and create a link
            self.trimDebug(debugLog, os.path.join(self.results_dir, 'debug_Trimmed'), os.path.join(self.results_dir, 'workflow_process.log'))

        else:
            print "ScriptStatus.csv does not exist in " +  self.raw_data_dir

        print "writing htmls..."
        self.write_html_block()


        if os.path.exists(os.path.join(self.raw_data_dir, 'pipetteUsage.json')):
            # make a local copy of the json file to remove Checksum
            f1 = open(os.path.join(self.raw_data_dir, 'pipetteUsage.json'), 'r')
            f2 = open(os.path.join(self.results_dir, 'pipetteUsage.json'), 'w')
            count = 0
            for line in f1:
                if count == 0:
                    f2.write('{')
                else:
                    if line.startswith( '}CheckSum:' ):
                        f2.write('}')
                    else:
                        f2.write(line)
                count = count + 1
            f1.close()
            f2.close()

            with open(os.path.join(self.results_dir, 'pipetteUsage.json' )) as f:  # table
                pipetteUsage = json.load(f)
            self.write_html(pipetteUsage)
        else:
            print "pipetteUsage.json does not exist in " +  os.path.join( self.raw_data_dir , 'pipetteUsage.json')

        

        print('Plugin complete.')

    def grepErr(self, debug_log, pattern):
        lines = []
        log = open(debug_log, "r")
        for line in log:
            if pattern in line: 
                lines.append(line) 
        log.close()
        return lines

    def trimDebug(self, debug, debug_Trimmed, workflow_process):
        found = False
        start_idx = []
        counter = 0
        with open(debug, 'rb') as fh:
            for line in fh:
                if "ExperimentStart" in line:
                    print "ExperimentStart -> " + str(counter) + "->" + line
                    start_idx.append(counter)
                counter = counter + 1

    
        f1 = open(debug, 'r')
        fline1 = f1.readlines()
        f1.close()
        f2 = open(debug_Trimmed, 'w')
        f2.writelines(fline1[start_idx[-1]::1])
        f2.close()
        f3 = open(workflow_process, 'w')
        with open(debug, 'rb') as fh:
            for line in fh:
                if 'Process:' in line:
                    f3.write(line)
                else:
                    pass
        f3.close()

        
    def createVacuumLogHTML(self, vacuum_log):
        header = []
        for row in range(len(vacuum_log)):
            onerow = vacuum_log.iloc[row]
            if row == 0:
                # parse header
                for i in range(len(onerow)):    
                    if (i % 2 == 0):
                        header.append(onerow[i])
        logTable = vacuum_log[range(1,len(vacuum_log.columns),2) ]
        logTable.columns= header
        print logTable
        logTable.to_html(self.results_dir + '/vacuum_log.html')
        
    '''
    ScriptStatus.csv in plugin
    '''
    def timePerformance(self, script_stats):
        ## create starttime 
        script_stats['time'] = script_stats['time'].replace('-', ' ', regex = True)
        script_stats['time'] = script_stats['time'].replace('_', '-', regex = True)

        script_stats['starttime'] = pd.to_datetime(script_stats['time'])
        # print pd.DataFrame(script_stats)
        events_stats = []
        script_started = script_stats.loc[script_stats[' status'].str.contains('started')]
        script_ended = script_stats.loc[script_stats[' status'].str.contains('completed')]

        for index, row in script_stats.iterrows():
            if 'started' in row[' status'] :
                dict1 = {}
                dict1['module'] = row[' module']
                dict1['submodule'] = row[' submodule']
                dict1['Task'] = dict1['submodule']
                dict1['Start'] = row['starttime']
                endtime = script_stats.loc[ (script_stats[' module'].str.contains(dict1['module'])) & (script_stats[' submodule'].str.contains(dict1['submodule'])) & (script_stats[' status'].str.contains('completed')) , 'starttime'].values
                if len(endtime) > 0: 
                    dict1['Finish'] = endtime[0]
                else:
                    dict1['Finish'] = dict1['Start']
                events_stats.append(dict1)
        # print pd.DataFrame(events_stats)
        events_stats_df = pd.DataFrame(events_stats)
        # fig = ff.create_gantt(pd.DataFrame(events_stats),title='Script Status',show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True)
        # py.iplot(fig, filename='Script Status')
        # if not os.path.exists('figures'):
        #     os.mkdir('figures')
        # offline.plot(fig, filename='Script_Status')

        module_stats = {}
        module_stats['library_preparation'] = events_stats_df['module']
        libPrep = events_stats_df.loc[ events_stats_df['module'].str.contains('library preparation') ]
        templating = events_stats_df.loc[ events_stats_df['module'].str.contains('templating') ]
        timesum_libPrep = 0
        for index, item in libPrep.iterrows():
            # print pd.to_timedelta(pd.to_datetime(item['Finish']) - pd.to_datetime(item['Start'])) / np.timedelta64(1, 'm')
            timesum_libPrep = timesum_libPrep + pd.to_timedelta(pd.to_datetime(item['Finish']) - pd.to_datetime(item['Start']))/np.timedelta64(1, 'm')
        timesum_templating = 0
        for index, item in templating.iterrows():
            # print pd.to_timedelta(pd.to_datetime(item['Finish']) - pd.to_datetime(item['Start'])) / np.timedelta64(1, 'm')
            timesum_templating = timesum_templating + pd.to_timedelta(pd.to_datetime(item['Finish']) - pd.to_datetime(item['Start']))/np.timedelta64(1, 'm')
        process = ('libprep', 'templating')
        time = [timesum_libPrep, timesum_templating]
        plt.figure(0)
        y_pos = np.arange(len(process))
        plt.barh(y_pos, time, align='center', height = 0.3, alpha=0.5)
        plt.yticks(y_pos, process)
        plt.xlabel('Tasks (minutes)')
        plt.title('Script Status')
        plt.savefig('script_stats.png')

    def plotTempCurrentVolt(self, temp_curr_volt):
        # PCR temp
        pd.to_datetime(temp_curr_volt['time'])
        plotValues(temp_curr_volt, 'time', 'PCRTemp', self.results_dir)
        plotValues(temp_curr_volt, 'time', 'PCRVoltage', self.results_dir)
        plotValues(temp_curr_volt, 'time', 'PCRCurrent', self.results_dir)
        plotValues(temp_curr_volt, 'time', 'Chip', self.results_dir)
        plotValues(temp_curr_volt, 'time', 'MagSepTemp', self.results_dir)
        plotValues(temp_curr_volt, 'time', 'ReagentBayTemp', self.results_dir)

    '''
    Vacuum log
            Table of even columns. (second page)
            Time to target, mean pressure (front page), pump duty cycle plots
    All lanes in one plot
    '''
    def plotVacuumLog(self, df, vacuum_head):
        cols = vacuum_head[0:-2:2]
        df = df[df.columns[1:-1:2]]
        df.columns = cols
        plotVocuum(df, 'TS', 'TimeToTarget', self.results_dir)
        plotVocuum(df, 'TS', 'MeanPressure', self.results_dir)
        plotVocuum(df, 'TS', 'PumpDutyCycle', self.results_dir)

    def write_html_block(self):
        """ Creates html and block html files """
        html = textwrap.dedent('''\
        <html>
        <head><title>LibPrepLog</title></head>
        <body>
        <object type="text/html" data="er52.html">
        <p>backup content</p>
        </object>

        <style type="text/css">div.scroll {max-height:800px; overflow-y:auto; overflow-x:auto; border: 0px solid black; padding: 5px 5px 0px 25px; float:left; }</style>
        <style type="text/css">div.plots {max-height:850px; overflow-y:auto; overflow-x:hidden; padding: 0px; }</style>
        <style type="text/css">tr.shade {background-color: #eee; }</style>
        <style type="text/css">td.top {vertical-align:top; }</style>
        <style>table.link {border: 1px solid black; border-collapse: collapse; padding: 0px; table-layout: fixed; text-align: center; vertical-align:middle; }</style>
        <style>th.link, td.link {border: 1px solid black; }</style>
        ''' )
        width = 33
        images = [['PCRTemp.png'], ['Chip.png']]
        # html += '<table cellspacing="0" width="100%%"><tr><td width="70%%">'
        # html += '<div class="plots">'
        html += '<table cellspacing="0" width="50%%">'
        for group in images:
            html += '<tr>'
            for pic in group:
                if not os.path.exists( os.path.join( self.results_dir , pic ) ):
                    pic = ''
                if pic == '':
                    html += '<td width="%s%%">&nbsp</td>' % width
                else:
                    html += '<td width="%s%%">%s</td>' % ( width , image_link( pic ))
            html += '</tr>'
        html += '</div></td></tr></table><br><hr>'
        html += '<h3><b>Debug (trimmed for last experiment):</b></h3>'
        html += '<table width="100%%" cellspacing="0" border="0">'
        html += '<td><a href="debug_Trimmed" target="_blank" >debug_Trimmed</a></td>'
        html += '</div></td></tr></table><br><hr>'
        html += '<h3><b>workflow process log:</b></h3>'
        html += '<table width="100%%" cellspacing="0" border="0">'
        html += '<td><a href="workflow_process.log" target="_blank" >workflow process log</a></td>'
        html += '</div></td></tr></table><br>'
        html += '</body></html>'

        with open( os.path.join( self.results_dir , 'libPrepLog_block.html' ) , 'w' ) as f:
            f.write( html )

    def write_html(self, pipetteUsage):
        """ Creates html and block html files """
        html = textwrap.dedent('''\
        <html>
        <head><title>LibPrepLog</title></head>
        <body>
        <style type="text/css">div.scroll {max-height:800px; overflow-y:auto; overflow-x:auto; border: 0px solid black; padding: 5px 5px 0px 25px; float:left; }</style>
        <style type="text/css">div.plots {max-height:850px; overflow-y:auto; overflow-x:auto; padding: 0px; }</style>
        <style type="text/css">tr.shade {background-color: #eee; }</style>
        <style type="text/css">td.top {vertical-align:top; }</style>
        <style>table.link {border: 1px solid black; border-collapse: collapse; padding: 0px; table-layout: fixed; text-align: center; vertical-align:middle; }</style>
        <style>th.link, td.link {border: 1px solid black; }</style>
        ''' )
        width = 33
        images = [['PCRCurrent.png', 'PCRVoltage.png', 'ReagentBayTemp.png', 'MagSepTemp.png'], ['PumpDutyCycle.png'], ['TimeToTarget.png', 'MeanPressure.png']]
        html += '<h3><b>Plots:</b></h3>'
        html += '<table cellspacing="0" width="100%%"><tr><td width="25%%">'
        html += '<div class="plots">'
        html += '<table cellspacing="0" width="100%%">'
        for group in images:
            html += '<tr>'
            for pic in group:
                if not os.path.exists( os.path.join( self.results_dir , pic ) ):
                    pic = ''
                if pic == '':
                    html += '<td width="%s%%">&nbsp</td>' % width
                else:
                    html += '<td width="%s%%">%s</td>' % ( width , image_link( pic ))
            html += '</tr>'
        html += '</div></td></tr></table><br><hr>'
        html += '<h3><b>Tube Vol History:</b></h3>'
        html += '<table width="100%%" cellspacing="0" border="0">'
        html += '<td><a href=" ' + self.raw_data_dir + '/tubeVolHistory.csv" download>TubeVolHistory </a></td>'
        html += '</div></td></tr></table><br><hr>'
        html += '<h3><b>Pipette Usage:</b></h3>'
        html += json2html.convert(json = pipetteUsage)
        
        with open( os.path.join( self.results_dir , 'libPrepLog.html' ) , 'w' ) as f:
            f.write( html )

if __name__ == "__main__":
    # os.system('/bin/bash -c export')
    PluginCLI()
