// html for picture container - note these are invisible and moved into position later
document.write('\
<div id="PF-frame" class="unselectable" style="page-break-inside:avoid;display:none">\
  <div id="PF-titlebar" class="grid-header">\
    <span id="PF-collapseDisplay" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Representation Plots</span>\
    <span class="PF-shy flyhelp" id="PF-displayLabel" style="padding-left:20px">Display:</span>\
    <select class="txtSelect PF-shy" id="PF-display">\
     <option value=2 selected="selected">Pass/Fail vs. Target G/C Content</option>\
     <option value=3>Pass/Fail vs. Target Length</option>\
     <option value=0>Representation vs. Target G/C Content</option>\
     <option value=1>Representation vs. Target Length</option>\
     <option value=4>Mean Target Reads vs. Pool</option>\
    </select>\
    <span id="PF-message" class="message"></span>\
  </div>\
  <div id="PF-displayArea" class="linkstable"></div>\
</div>\
');

$(function () {

  var disableTitleBar = false;

  // check placer element exists
  if( !$('#PictureFrame').length ) return;

  // minimum sizes for chart widget
  var def_minWidth = 620;
  var def_minHeight = 200;

  // configure widget size and file used from placement div attributes
  var gccovFile = $("#PictureFrame").attr("gccovfile");
  if( gccovFile === undefined ) gccovFile = '';
  var lencovFile = $("#PictureFrame").attr("lencovfile");
  if( lencovFile === undefined ) lencovFile = '';
  var fedoraFile = $("#PictureFrame").attr("fedorafile");
  if( fedoraFile === undefined ) fedoraFile = '';
  var fedlenFile = $("#PictureFrame").attr("fedlenfile");
  if( fedlenFile === undefined ) fedlenFile = '';
  var poolcovFile = $("#PictureFrame").attr("poolcovfile");
  if( poolcovFile === undefined ) poolcovFile = '';

  var startCollapsed = $("#PictureFrame").attr("collapse");
  startCollapsed = (startCollapsed != undefined);


  if( gccovFile === '' && lencovFile === '' && fedoraFile === '' && fedlenFile === '' && poolcovFile === '' ) {
    // in tradition of HTML, an empty div results in no complaints
    //alert("ERROR on page: PictureFrame widget requires a file attribute set.");
    $('#PictureFrame').hide();
    return;
  }
  if( gccovFile === '' ) $("#PF-display option[value=0]").hide();
  if( lencovFile === '' ) $("#PF-display option[value=1]").hide();
  if( fedoraFile === '' ) $("#PF-display option[value=2]").hide();
  if( fedlenFile === '' ) $("#PF-display option[value=3]").hide();
  if( poolcovFile === '' ) $("#PF-display option[value=4]").hide();

  var tmp = $('#PictureFrame').width();
  if( tmp < def_minWidth ) tmp = def_minWidth;
  $("#PF-displayArea").width(tmp);
  tmp = $('#PictureFrame').height();
  if( tmp < def_minHeight ) tmp = def_minHeight;
  $("#PF-displayArea").height(tmp-36);
  $("#PictureFrame").css('height','auto');

  $("#PF-frame").appendTo('#PictureFrame');
  $("#PF-frame").show();

  var resiz_def = {
    alsoResize: "#PF-displayArea",
    minWidth:def_minWidth,
    minHeight:def_minHeight,
    handles:"e,s,se"
  };
  $('#PF-frame').resizable(resiz_def);

  $("#PF-collapseDisplay").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#PF-displayArea').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#PF-frame').resizable('destroy');
      $('.PF-shy').fadeOut(400);
      $('#PF-displayArea').slideUp('slow',function(){
        $('#PF-titlebar').css("border","1px solid grey");
      });
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('.PF-shy').fadeIn(400);
      $('#PF-titlebar').css("border-bottom","0");
      $('#PF-displayArea').slideDown('slow',function(){
        $('#PF-frame').resizable(resiz_def);
      });
    }
    $("#PF-frame").css('height','auto');
  });

  $('#PF-display').change( function() {
    var sel = parseInt(this.value);
    if( sel == 0 ) {
      setFramePicture(gccovFile);
    } else if( sel == 1 ) {
      setFramePicture(lencovFile);
    } else if( sel == 2 ) {
      setFramePicture(fedoraFile);
    } else if( sel == 3 ) {
      setFramePicture(fedlenFile);
    } else if( sel == 4 ) {
      setFramePicture(poolcovFile);
    }
  });

  var curPicFile = '';

  function setFramePicture(picFile) {
    curPicFile = picFile;
    $('#PF-displayArea').html('<div class="imageplot" style="height:100%;width:100%"><img style="height:100%;width:100%" src="'+picFile+'"/></div>');
  }

  function customizeChart() {
    // add fly-over help to controls here in case need to customize for chart data
    $("#PF-displayLabel").attr( "title",
      "Select a static representation plot for display." );
  }

  if( fedoraFile != '' ) {
    setFramePicture(fedoraFile);
  } else if( gccovFile != '' ) {
    setFramePicture(gccovFile);
  } else if( fedlenFile != '' ) {
    setFramePicture(fedlenFile);
  } else if( lencovFile != '' ) {
    setFramePicture(lencovFile);
  } else if( poolcovFile != '' ) {
    setFramePicture(poolcovFile);
  }
  customizeChart();

  if( startCollapsed ) {
    $("#PF-collapseDisplay").attr("class","ui-icon ui-icon-triangle-1-s");
    $("#PF-collapseDisplay").attr("title","Expand view");
    $('#PF-frame').resizable('destroy');
    $('.PF-shy').hide();
    $('#PF-displayArea').hide();
    $('#PF-titlebar').css("border","1px solid grey");
  }

});
