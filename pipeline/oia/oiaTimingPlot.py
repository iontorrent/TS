#/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
import pandas as pd
import datetime
import dateutil
import matplotlib.dates as dates
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("cpu_util.log",
names = ['time', 'j1', 'j2', 'usr', 'nice', 'sys', 'iowait', 'irq', 'soft', 'steal', 'guest', 'gnice', 'idle'],
delim_whitespace=True, header=None)

data = df.T.to_dict().values()#export the data frame to a python dictionary
x_axis = np.zeros(len(data),dtype='datetime64[s]')
y_axis_idle = np.zeros(len(data))
y_axis_idle_smoothed = np.zeros(len(data))
y_axis_usr = np.zeros(len(data))
y_axis_usr_smoothed = np.zeros(len(data))
y_axis_nice = np.zeros(len(data))
y_axis_nice_smoothed = np.zeros(len(data))
y_axis_sys = np.zeros(len(data))
y_axis_sys_smoothed = np.zeros(len(data))
now = datetime.date.today().isoformat() + ' '
span=5
span_gpu=10

for key in range(0,len(data)):
    datekey = now + data[key]['time']
    x_axis[key] = np.datetime64(datekey)
    y_axis_idle[key] = int(data[key]['idle'])
    y_axis_usr[key] = int(data[key]['usr'])
    y_axis_nice[key] = int(data[key]['nice'])
    y_axis_sys[key] = int(data[key]['sys'])


# now, read in the gpu data
df=pd.read_csv("gpu_util.log", names=['systemtime', 'percent'],sep=',',parse_dates=[0]) #or:, infer_datetime_format=True)
data2=df.T.to_dict().values()#export the data frame to a python dictionary
x_axis_gpu = np.zeros(len(data2),dtype='datetime64[s]')
y_axis_gpu = np.zeros(len(data))
y_axis_gpu_smoothed = np.zeros(len(data))

for key in range(0,len(data)):
    x_axis_gpu[key] = np.datetime64((data2[key]['systemtime']))
    if key < len(data2):
        y_axis_gpu[key] = int((data2[key]['percent'].replace(" ","").replace("%","")))
    else:
        y_axis_gpu[key] = 0	

#print x_axis[0]
#print x_axis_gpu[0]
#print x_axis[len(x_axis)-1]
#print x_axis_gpu[len(x_axis_gpu)-1]

# smooth the data
if len(data) > span:
    for key in range(span,len(data)-span):
        sum_gpu = 0
        for key2 in range(key-span,key+span):
            sum_gpu += y_axis_gpu[key2];
        y_axis_gpu_smoothed[key] = sum_gpu/(2*span)

    for key in range(span,len(data)-span):
        sum_idle = sum_usr = sum_nice = sum_sys = 0
        for key2 in range(key-span,key+span):
            sum_idle += y_axis_idle[key2];
            sum_usr  += y_axis_usr[key2];
            sum_nice += y_axis_nice[key2];
            sum_sys  += y_axis_sys[key2];
        y_axis_idle_smoothed[key] = sum_idle/(2*span)
        y_axis_usr_smoothed[key]  = sum_usr/(2*span)
        y_axis_nice_smoothed[key] = sum_nice/(2*span)
        y_axis_sys_smoothed[key]  = sum_sys/(2*span)

s = data
wl=0.6
fsz=8
fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fsz)

#xstart, xend = ax.get_xlim()
#xtickvals = 
#ax.xaxis.set_ticks(xtickvals)

plt.plot(x_axis, y_axis_usr,  '#be4b48', linewidth=wl, label='% usr')
plt.plot(x_axis, y_axis_nice, '#98b954', linewidth=wl, label='% nice')
plt.plot(x_axis, y_axis_sys,  '#7d60a0', linewidth=wl, label='% sys')
plt.plot(x_axis, y_axis_idle, '#46aac5', linewidth=wl, label='% idle')
plt.plot(x_axis, y_axis_gpu,  '#000000', linewidth=wl, label='% gpu')
plt.legend(loc='right', bbox_to_anchor=(1.25, 0.5), fontsize=fsz)
plt.savefig("oiaTimingRaw.png")
plt.clf()
wl=1.0
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fsz)

plt.plot(x_axis, y_axis_usr_smoothed,  '#be4b48', linewidth=wl, label='% usr')
plt.plot(x_axis, y_axis_nice_smoothed, '#98b954', linewidth=wl, label='% nice')
plt.plot(x_axis, y_axis_sys_smoothed,  '#7d60a0', linewidth=wl, label='% sys')
plt.plot(x_axis, y_axis_idle_smoothed, '#46aac5', linewidth=wl, label='% idle')
plt.plot(x_axis, y_axis_gpu_smoothed,  '#000000', linewidth=0.4, label='% gpu')
plt.legend(loc='right', bbox_to_anchor=(1.25, 0.5), fontsize=fsz)
plt.savefig("oiaTiming.png")

