
from ion.plugin import *
import os
import csv
import subprocess
import numpy as np
import traceback
from utils import *


def parse_explog(analysisRoot):
    explog_file = '%s/explog_final.txt'%analysisRoot
    if not os.path.isfile(explog_file):
        explog_file = '%s/explog.txt'%analysisRoot
    with open(explog_file, 'r') as data_file:
        explog = {s[0]:':'.join(s[1:len(s)]).strip().replace(' ','_') for s in csv.reader(data_file, delimiter=':') if len(s)>=2}
    return explog

def ActiveLanes(analysisRoot):
    '''
    Return True/False for each lane number
    '''
    p = parse_explog(analysisRoot)
    activeLane = dict()
    for lane in range(1,5):
        activeLane[lane] = p['LanesActive%d'%lane] == 'yes'
    return activeLane


class True_loading(IonPlugin):
    """ Remove empty wells misclassified as  wells and recalculate statistics"""
    version = "1.3"
    allow_autorun = True # if true, no additional user input
    runtypes = [ RunType.THUMB, RunType.FULLCHIP, RunType.COMPOSITE ]
    #runtypes = [ RunType.THUMB, RunType.FULLCHIP ]
    
    def launch(self):
        """ main """

        log("")

        import json
        # start_json = getattr(self, 'startpluginjson', None)
        # if not start_json:
        #     try:
        #         with open('startplugin.json', 'r') as fh:
        #             start_json = json.load(fh)
        #     except:
        #         self.log.error("Error reading start plugin json")
        
        self.results_dir = os.environ['TSP_FILEPATH_PLUGIN_DIR']
        self.analysis_dir = os.environ['ANALYSIS_DIR'] 
        self.sigproc_dir = os.environ["SIGPROC_DIR"]
        self.plugin_dir = os.environ["DIRNAME"]

        self.run_type = os.environ["TSP_RUNTYPE"]
        self.chip_type = os.environ["TSP_CHIPTYPE"]

        # THUMBNAIL? 
        print self.chip_type
        if self.run_type=='thumbnail' or '318' in self.chip_type:
            # run plotting script
            log("Plotting...")
            command = "Rscript %s/True_loading.R " % self.plugin_dir
            command += "%s " % self.analysis_dir
            command += "%s/ " % self.results_dir
            command += "%s " % self.plugin_dir
            run( command )    
            log("")
        else:
            # run plotting script
            log("Plotting...")
            command = "Rscript %s/True_loading_fullchip.R " % self.plugin_dir
            command += "%s " % self.analysis_dir
            command += "%s/ " % self.results_dir
            command += "%s " % self.plugin_dir
            command += "%s " % self.plugin_dir
            run( command )    
            log("")
            
        try:
            # save block.html
            with open(self.results_dir+'/results.json') as data_file:
                results = json.load(data_file)
            html_fname = "%s/True_loading_block.html" % self.results_dir
            with open(html_fname, 'w') as html_out:
                html_out.write('<body>\n')
                html_out.write('Exclude empty wells that are misclassified as bead wells and calculate filter statistics <br>')
	         
                html_out.write('<table><tr><td align="center">')
                html_out.write('<img src="copycount_true.png"/>')
                html_out.write('</td><td align="center">')
                html_out.write('<img src="copycount_pipeline.png"/>')
                html_out.write('</td></tr>')
                
                if not (self.run_type=='thumbnail' or '318' in self.chip_type):
			activeLane = ActiveLanes(self.analysis_dir)
			html_out.write('<tr>')
			tableCount = 0
			for lane in range(1,5):
			     if activeLane[lane]:
				if tableCount == 2:
				    html_out.write('</tr>  <td></td> <tr>')
				l = 'lane%d'%lane
				html_out.write('<td align = "center">')
				html_out.write('<table width="50%" border="1">')
				html_out.write('<tr><td><b>%s</b></td><td><b>%s</b></td><td><b>%s</b></td></tr>'%(l,'True', 'Pipeline'))
				html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Loading',results[l]['true']['loadingPercent'][0], results[l]['apparent']['loadingPercent'][0]))
				html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Polyconal',results[l]['true']['polyClonalPercent'][0], results[l]['apparent']['polyClonalPercent'][0]))
				html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Low quality',results[l]['true']['lowQualityPercent'][0], results[l]['apparent']['lowQualityPercent'][0]))
				html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Usable',results[l]['true']['usablePercent'][0], results[l]['apparent']['usablePercent'][0]))
				html_out.write('</table>')
				html_out.write('</td>')
				tableCount += 1
			html_out.write('</tr>')         
		

                html_out.write('<tr><td align = "center">')
                html_out.write('<table width="50%" border="1">')
                html_out.write('<tr><td><b>%s</b></td><td><b>%s</b></td><td><b>%s</b></td></tr>'%('full chip','True', 'Pipeline'))
                html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Loading',results['true']['loadingPercent'][0], results['apparent']['loadingPercent'][0]))
                html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Polyconal',results['true']['polyClonalPercent'][0], results['apparent']['polyClonalPercent'][0]))
                html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Low quality',results['true']['lowQualityPercent'][0], results['apparent']['lowQualityPercent'][0]))
                html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Usable',results['true']['usablePercent'][0], results['apparent']['usablePercent'][0]))
                html_out.write('</table>')
                html_out.write('</td>')
                
            
                # true mean/sd table
                html_out.write('<td align = "center">')
                html_out.write('<b>True copy count mean and standard deviation:</b>')
                html_out.write('<br>')
                html_out.write('<table width="50%" border="1">')
                html_out.write('<tr><td>%s</td><td><b>%s</b></td><td><b>%s</b></td></tr>'%('','Mean', 'SD'))
                typeName = ['Library', 'Usable', 'Polyclonal', 'Low quality', 'Bad key']
                typeTag = ['lib', 'passFilter', 'poly', 'lowQuality', 'badKey']
                for name, tag in zip(typeName, typeTag):
                    html_out.write('<tr><td>%s</td><td>%1.2f</td><td>%1.2f</td></tr>'%(name ,results['copyCount']['trueMean'][tag][0], results['copyCount']['trueSD'][tag][0]))
                html_out.write('</table>')
                html_out.write('</td></tr>')

                # spatial true loading and usable plots
                html_out.write('<tr><td  align = "center">')
                html_out.write('<img src="spatial_true_plot_loading.png"/></a>')
                html_out.write('<br><a href="true_loading_per_block.txt">True loading values</a>')
                html_out.write('</td>')
                html_out.write('<td align ="center">')
                html_out.write('<img src="spatial_true_plot_usable.png"/></a>')
                html_out.write('<br><a href="true_usable_per_block.txt">True usable values</a>')
                html_out.write('</td></tr>')

                html_out.write('</table></body>')

            # True spatial plot
            html_fname = "%s/All_spatial_true_plots.html" % self.results_dir
            with open(html_fname, 'w') as html_out:
                html_out.write('<body>\n')
                html_out.write('<table><tr><td align="center">')
                html_out.write('<img src="spatial_true_plot_addressable.png"/>')
                html_out.write('</td><td align="center">')
                html_out.write('<img src="spatial_true_plot_loading.png"/>')
                html_out.write('</td></tr>')

                html_out.write('<tr><td align="center">')
                html_out.write('<img src="spatial_true_plot_usable.png"/>')
                html_out.write('</td><td align="center">')
                html_out.write('<img src="spatial_true_plot_poly.png"/>')
                html_out.write('</td></tr>')

                html_out.write('<tr><td align="center">')
                html_out.write('<img src="spatial_true_plot_lowQuality.png"/>')
                html_out.write('</td><td align="center">')
                html_out.write('<img src="spatial_true_plot_badKey.png"/>')
                html_out.write('</td></tr>')


                html_out.write('</table></body>\n')

            # additional plot html
            if self.run_type != 'thumbnail':
                html_fname = "%s/All_true_per_block_plots.html" % self.results_dir
                with open(html_fname, 'w') as html_out:
                    html_out.write('<body>\n')
                    html_out.write('<table><tr><td align="center">')
                    html_out.write('<img src="true_loading_per_block.png"/>')
                    html_out.write('</td><td align="center">')
                    html_out.write('<img src="true_usable_per_block.png"/>')
                    html_out.write('</td></tr>')
                    html_out.write('<tr><td align="center">')
                    html_out.write('<img src="true_poly_per_block.png"/>')
                    html_out.write('</td><td align="center">')
                    html_out.write('<img src="true_lowQuality_per_block.png"/>')
                    html_out.write('</td></tr>')
                    html_out.write('<tr><td align="center">')
                    html_out.write('<img src="copycount_true.png"/>')
                    html_out.write('</td>')
                    html_out.write('<td align="center">')
                    html_out.write('<table width="50%" border="1">')
                    html_out.write('<tr><td>%s</td><td><b>%s</b></td><td><b>%s</b></td></tr>'%('','True', 'Pipeline'))
                    html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Loading',results['true']['loadingPercent'][0], results['apparent']['loadingPercent'][0]))
                    html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Polyconal',results['true']['polyClonalPercent'][0], results['apparent']['polyClonalPercent'][0]))
                    html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Low quality',results['true']['lowQualityPercent'][0], results['apparent']['lowQualityPercent'][0]))
                    html_out.write('<tr><td>%s</td><td>%1.1f%%</td><td>%1.1f%%</td></tr>'%('Usable',results['true']['usablePercent'][0], results['apparent']['usablePercent'][0]))
                    html_out.write('</table>')
                    html_out.write('</tr>')
                    html_out.write('</table></body>\n')

                # True histograms per block
                html_fname = "%s/All_true_histograms_per_block_plot.html" % self.results_dir
                with open(html_fname, 'w') as html_out:
                    blocks = [i for i in results['truePerRegion']]
                    X = [ int(b.split('_')[1].replace('X','')) for b in blocks]
                    Y = [ int(b.split('_')[2].replace('Y','')) for b in blocks]
                    html_out.write('<body>\n')
                    html_out.write('<table>')
                    for y in sorted(np.unique(Y), reverse=True):
                        html_out.write('<tr>')
                        for x in sorted(np.unique(X)):
                            html_out.write('<td align="center"> <a href="block_X%d_Y%d/copycount_true.png"><img src="block_X%d_Y%d/copycount_true.png" width="300" height="200"/> </td>'%(x,y, x, y) )
                        html_out.write('</tr>')
                    html_out.write('</table>')
                    html_out.write('</body>\n')
        except Exception, e:
            self.log.error(traceback.print_exc())
            return False
        return True

if __name__ == "__main__": PluginCLI()
