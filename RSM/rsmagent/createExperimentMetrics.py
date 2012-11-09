#!/usr/bin/python

# Creates a file named TSExperiment-UUID.txt that contains 
# metrics from the Results of an experiment. 
#
# The primary key (id) of iondb.rundb.Results primary key is
# expected in argv[1].


import sys
import os
import uuid

sys.path.append('/opt/ion/')
sys.path.append("/opt/ion/iondb/")
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'


import rundb.models

if __name__=="__main__":

    # the following would be passed as an arg to the script
#    rr = rundb.models.Results.objects.all().filter(resultsName="b")[0]
#    resultId = rr.id
#    print str(resultId)
#    resultId = rr.id

    # results primary key should be passed to this script
    resultId = sys.argv[1];

    r = rundb.models.Results.objects.get(id=resultId)
    if not r:
        sys.exit(1) 

    metrics = [ ]

    # information from explog
    e = r.experiment
    explog = e.log
    keys = [ 
              'serial_number',
              'run_number',
              'chipbarcode',
              'chiptype',
              'seqbarcode',
              'flows',
              'cycles',
              'gain',
              'noise',
              'cal_chip_high_low_inrange'
             ]
    for k in keys:
        v = explog.get(k)
        if v:
            metrics.append(k + ':' + str(v))

    # information from libmetrics
    keys = [ 
              'sysSNR',
              'aveKeyCounts',
         ]
    for k in keys:
        try:
            # there should be only one libmetrics in the set
            v = r.libmetrics_set.values()[0][k]
            if v:
                metrics.append(k + ':' + str(v))
        except:
            pass

    # bail if no metrics to report (should not happen)
    if not metrics:
        sys.exit(0)  
        
    # write out the metrics
    x = uuid.uuid1()
    fname = os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])),
                'TSexperiment-' + str(x) + '.txt') 
    f = open(fname, 'w' )
#    f = open('/tmp/TSexperiment-' + str(x) + '.txt', 'w' )
    if not f:
        sys.exit(1)  

    try:
        f.write("\n".join(metrics))
        f.write("\n")
    finally:
        f.close()
        os.chmod(fname, 0666)
        
