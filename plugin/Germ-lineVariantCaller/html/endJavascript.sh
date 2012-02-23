#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


print_html_end_javascript()
{
echo "            <!-- zebra stripe the tables -->
			<script type=\"text/javascript\">
				 
			 \$(document).ready(function(){

			   \$('h2').append('<a href=\"javascript:;\" class=\"expandCollapseButton\" title=\"Collapse Section\"></a>');

				\$('.expandCollapseButton').toggle(function() {
					if ( \$(this).attr('title') == 'Collapse Section'){
						\$(this).css('background-position','right top');
						\$(this).attr('title','Expand Section');
					}else{
						\$(this).css('background-position','left top');
						\$(this).attr('title','Collapse Section');
					}
				}, function() {
					if ( \$(this).attr('title') == 'Expand Section'){
						\$(this).css('background-position','left top');
						\$(this).attr('title','Collapse Section');
					}else{
						\$(this).css('background-position','right top');
						\$(this).attr('title','Expand Section');
					}
				});
				
				\$('.expandCollapseButton').click(function(event){
					\$(event.target).parent().parent().toggleClass('small');
					\$(event.target).parent().next().slideToggle();
				});

				\$('#tf .expandCollapseButton').css('background-position','right top');\$('#tf .expandCollapseButton').attr('title','Expand Section');\$('#tf').parent().toggleClass('small');\$('#tf').next().toggle();
				//start overlay

				\$(\".heading tbody tr\").mouseover(
						function(){
							\$(this).addClass(\"table_hover\");
							
				}).mouseout(
						function(){
							\$(this).removeClass(\"table_hover\");
				});

				\$(\".noheading tbody tr\").mouseover(
						function(){
							\$(this).addClass(\"table_hover\");
							
				}).mouseout(
						function(){
							\$(this).removeClass(\"table_hover\");
				});
				
				\$(\".heading tr:odd\").addClass(\"zebra\");
				\$(\".noheading tr:odd\").addClass(\"zebra\");";

if [[ ${PLUGIN_OUT_TOTAL_DISPLAYED} ]]; then
	echo "\$('#datatable').dataTable( {";
if [ ${PLUGIN_OUT_TOTAL_DISPLAYED} -le 10 ]; then
	echo "					\"aLengthMenu\": [[${PLUGIN_OUT_TOTAL_DISPLAYED}], [${PLUGIN_OUT_TOTAL_DISPLAYED}]],";
elif [ ${PLUGIN_OUT_TOTAL_DISPLAYED} -le 25 ]; then
	echo "					\"aLengthMenu\": [[10, ${PLUGIN_OUT_TOTAL_DISPLAYED}], [10, ${PLUGIN_OUT_TOTAL_DISPLAYED}]],";
elif [ ${PLUGIN_OUT_TOTAL_DISPLAYED} -le 50 ]; then
	echo "					\"aLengthMenu\": [[10, 25, ${PLUGIN_OUT_TOTAL_DISPLAYED}], [10, 25, ${PLUGIN_OUT_TOTAL_DISPLAYED}]],";
else
	echo "					\"aLengthMenu\": [[10, 25, 50, ${PLUGIN_OUT_TOTAL_DISPLAYED}], [10, 25, 50, ${PLUGIN_OUT_TOTAL_DISPLAYED}]],";
fi
echo \
"					\"sDom\": '<\"top\"fl>rt<\"bottom\"ip><\"spacer\">', 
					\"sScrollX\": \"100%\",
					\"sScrollXInner\": \"110%\",
					\"bScrollCollapse\": true,
					\"sPaginationType\": \"full_numbers\"
				} );";	
			fi
echo "			});
			 
		  </script>
";
}
