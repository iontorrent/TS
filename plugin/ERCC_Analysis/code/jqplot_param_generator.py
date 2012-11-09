# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import division
from proc_ercc import dose_response
from math import log
import operator

def color_code(mean_mapq):
  if (mean_mapq < 60):
    return '#FF0000'
  elif (mean_mapq < 70):
    return '#FF8800'
  elif (mean_mapq < 80):
    return '#88FF00'
  elif (mean_mapq < 90):
    return '#008888'
  else:
    return '#0000FF'

def chart_series_params(counts,transcript_sizes,transcript_names, transcript_mapqs, ercc_conc):
  transcript_names_list = []
  transcript_sizes_list = []
  transcript_images_list = []
  transcript_counts_list = []
  transcript_master_list = []
  transcript_mapq_list = []
  max_transcript_counts = 0
  for ercc, count in counts.iteritems():
    transcript_size = int(transcript_sizes[transcript_names.index(str(ercc))])
    transcript_long_name = str(ercc) + '(' + str(transcript_size) + 'bp)'
    transcript_mapq = transcript_mapqs[str(ercc)]
    transcript_master_list.append((transcript_long_name,'<img class="coverage_plot" src="'+ercc+'.png">',count,transcript_size,transcript_mapq))
    if (count > max_transcript_counts):
      max_transcript_counts = count
  transcript_master_list.sort() #sorts by ercc, alphabetically
  series_options_template = """{
                                        label: '%(name)s',
                                        showLine:false,
                                        markerOptions: {
                                                size: %(counts_size)d,
                                                color: '%(mapq)s'
                                        },
                                        highlighter: {
                                                tooltipLocation: '%(highlighter_tooltiplocation)s'
                                        }
                                },"""
  ercc_points_template = "[[%(ercc_conc)2.2f,%(counts)2.2f,'%(name)s',%(reads)d,%(mean_mapq)d,'%(plot)s']],"
  series_options = 'series:['
  ercc_points = ''
  template_options = {}
  for transcript,ercc_transcript_conc in zip(transcript_master_list,ercc_conc):
    transcript_names_list.append(transcript[0])
    transcript_images_list.append(transcript[1])
    transcript_counts_list.append(transcript[2])
    transcript_sizes_list.append(transcript[3])
    template_options['ercc_conc'] = log(ercc_transcript_conc[1],2)
    if (template_options['ercc_conc'] < 11):
      template_options['highlighter_tooltiplocation'] = 'se'
    else:
      template_options['highlighter_tooltiplocation'] = 's'
    template_options['reads'] = transcript[2]
    template_options['counts'] = log(transcript[2],2)
    template_options['name'] = transcript[0]
    template_options['plot'] = transcript[1]
    if ((transcript[2]>0) and (max_transcript_counts>0) and (log(max_transcript_counts,2)>0) ):    
      template_options['counts_size'] = 5+(10*(log(transcript[2],2)/log(max_transcript_counts,2))) #from 5 to 15 in counts
    else:
      template_options['counts_size'] = 5
    template_options['mean_mapq'] = transcript[4]
    template_options['mapq'] = color_code(transcript[4]) 
    ercc_points += (ercc_points_template % template_options)
    series_options += (series_options_template % template_options)
  series_options += "{label: 'trendline',showMarker:false}],"  
  return transcript_names_list, transcript_images_list, transcript_counts_list, transcript_sizes_list, series_options, ercc_points

def generate_trendline_points(dr):
  second_trendline_point_x = 0
  for x in dr[0]:
    if (x > second_trendline_point_x):
      second_trendline_point_x = x

  second_trendline_point_y = (dr[2] * second_trendline_point_x) + dr[3]

  trendline_points = '['
  trendline_points += '[0,'+str(dr[3])+'],'
  trendline_points += '['+str(second_trendline_point_x)+','+str(second_trendline_point_y)+']'
  trendline_points += ']'

  return trendline_points
 
def generate_color_legend():
  rgb_code = color_code(99)
  color_legend = '<tr><td>more than 90</td><td>=</td><td style="background-color:'+rgb_code+'"></tr>'
  for mean_mapq in range(90,50,-10):
    rgb_code = color_code(mean_mapq-1)
    color_legend += '<tr><td>less than '+str(mean_mapq)+'</td><td>=</td><td style="background-color:'+rgb_code+'"></tr>'
  return color_legend

