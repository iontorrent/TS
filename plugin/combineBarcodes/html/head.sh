#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

print_html_head()
{
echo "<head>
<script type=\"text/javascript\" src=\"/site_media/jquery/js/jquery-1.6.1.min.js\"></script>
<script type=\"text/javascript\" src=\"/site_media/jquery/js/jquery-ui-1.8.13.min.js\"></script>
<script type=\"text/javascript\" src=\"/site_media/jquery/js/tipTipX/jquery.tipTipX.js\"></script>";

echo "
<link rel=\"stylesheet\" type=\"text/css\" href=\"/site_media/stylesheet.css\"/>
<link type=\"text/css\" href=\"/site_media/jquery/css/aristo/jquery-ui-1.8.7.custom.css\" rel=\"stylesheet\" />
<link href=\"/site_media/jquery/js/tipTipX/jquery.tipTipX.css\" rel=\"stylesheet\" type=\"text/css\" />
<link rel=\"stylesheet\" type=\"text/css\" href=\"/site_media/report.css\" media=\"screen\" />";

local CSS;
for CSS in `ls -1 ${TSP_FILEPATH_PLUGIN_DIR}/css/*.css` 
do
  CSS=`echo ${CSS} | sed -e 's_.*/__g'`;
  echo "<link rel=\"stylesheet\" type=\"text/css\" href=\"css/${CSS}\" />";
done

echo \
"<script type=\"text/javascript\">
  \$(function() {
    \$().tipTipDefaults({ delay : 0 });
    \$(\".tip\").tipTip({ position : \"bottom\" });
  });
</script>
<script type=\"text/javascript\">
  \$(function() {
    \$( \"#accordion\").accordion();
  });
</script>";
if [ "$1" -gt 0 ]; then
    echo "<META HTTP-EQUIV=\"refresh\" CONTENT=\"${1}\">"
fi
echo "</head>";
}
