#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

write_html_report_list()
{
echo \
" <div>
  <style type=text/css>
   th {text-align:center}
  </style>
  <table class="noheading" style="table-layout:fixed">
   <thead>
    <tr>
      <th style=\"width:370px !important\">Report Name</th>
      <th style=\"width:170px !important\">Project</th>
      <th style=\"width:100px !important\">AQ17 Reads</th>
      <th style=\"width:100px !important\">TMAP Version</th>
      <th style=\"width:110px !important\">Analysis Date</th>
     </tr>
    </thead>
   <tbody>";
if [ -f "$1" ]; then
    cat "$1" 2> /dev/null;
fi
echo \
"   </tbody>
  </table>
 </div>
 <br>";
}
