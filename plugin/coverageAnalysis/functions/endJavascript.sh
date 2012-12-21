#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

print_html_end_javascript()
{
echo "
<script type=\"text/javascript\">
\$(document).ready(function(){
  \$('h2').prepend('<a href=\"javascript:;\" class=\"expandCollapseButton\" title=\"Collapse Section\"></a>');
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

  \$('#tf .expandCollapseButton').css('background-position','right top');
  \$('#tf .expandCollapseButton').attr('title','Expand Section');
  \$('#tf').parent().toggleClass('small');
  \$('#tf').next().toggle();

  \$(\".heading tbody tr\").mouseover(function(){
      \$(this).addClass(\"table_hover\");
   }).mouseout(function(){
      \$(this).removeClass(\"table_hover\");
   });

  \$(\".noheading tbody tr\").mouseover(function(){
      \$(this).addClass(\"table_hover\");
   }).mouseout(function(){
      \$(this).removeClass(\"table_hover\");
   });
    
  \$(\".heading tr:odd\").addClass(\"zebra\");
  \$(\".noheading tr:odd\").addClass(\"zebra\");
});
</script>";
}
