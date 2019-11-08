// html for chart container and filters bar - note these are invisible and moved into position later
document.write('\
<div id="RC-chart" class="unselectable" style="border:2px solid #666;display:none">\
  <div id="RC-titlebar" class="grid-header" style="min-height:24px;border:0">\
    <span id="RC-collapsePlot" style="float:left" class="ui-icon ui-icon-triangle-1-n" title="Collapse View"></span>\
    <span class="table-title" style="float:none">Reference Coverage Chart</span>\
    <span class="RC-shy flyhelp" id="RC-plotLabel" style="padding-left:20px">Plot:</span>\
    <select class="RC-selectParam txtSelect RC-shy" id="barAxis">\
     <option value=0 selected="selected">Total Base Reads</option>\
     <option value=1>Strand Base Reads</option>\
    </select>\
    <span class="RC-shy flyhelp" id="RC-overlayLabel" style="padding-left:10px">Overlay:</span>\
    <select class="RC-selectParam txtSelect RC-shy" id="overPlot">\
     <option value=0 selected="selected">None</option>\
     <option value=1>On-Target</option>\
     <option value=2>Strand Bias</option>\
    </select>\
    <input class="RC-shy" id="RC-unzoomToggle" type="button" value="Zoom In" style="margin-left:10px;width:70px">\
    <span id="RC-toggleControls" style="float:right" class="RC-shy ui-icon ui-icon-search" title="Show/Hide the view options panel"></span>\
    <span id="RC-help" style="float:right;margin-left:0;margin-right:0" class="RC-shy ui-icon ui-icon-help"></span>\
    <span id="RC-message" class="message"></span>\
  </div>\
  <div id="RC-noncanvas" style="background:#EEF;border-top:2px solid #666">\
   <div id="RC-plotspace" style="padding:4px">\
    <div id="RC-slider" class="grid-microslider" style="display:none"></div>\
    <div id="RC-placeholder" style="width:100%"></div>\
   </div>\
  </div>\
</div>\
<div id="RC-controlpanel" class="filter-panel" style="display:none">\
  <table><tr>\
    <td class="nwrap">View Options:</td>\
    <td class="nwrap" id="RC-autoZoomControl"><span class="flyhelp" id="RC-autoZoomLabel">Automatic Zoom</span>:\
      <input type="checkbox" id="RC-autoZoom" checked="unchecked"></td>\
    <td class="nwrap" id="RC-offScaleOutlierControl" style="display:none">\
      <span class="flyhelp" id="RC-offScaleOutlierLabel">Plot Outlier Off-scale</span>:\
      <input type="checkbox" id="RC-offScaleOutlier" checked="unchecked"></td>\
    <td class="nwrap"><span class="flyhelp" id="RC-showLegendLabel">Show Legend</span>:\
      <input type="checkbox" id="RC-showLegend" checked="checked"></td>\
    <td class="nwrap" id="RC-showTargetsOption"><span class="flyhelp" id="RC-showTargetsLabel">Show Targets</span>:\
      <input type="checkbox" id="RC-showTargets" checked="checked"></td>\
    <td class="nwrap"><span class="flyhelp" id="RC-numPointsLabel">Bars/Points</span>:\
      <input type="text" class="numSearch" id="RC-numPoints" value=200 size=4>&nbsp;<span id="RC-numBars"></span></td>\
    <td><input id="RC-export" type="button" value="Export"></td>\
  </tr></table>\
  <table><tr>\
    <td class="nwrap">Reference Region:</td>\
    <td class="nwrap"><span class="flyhelp" id="RC-filterChromLabel">Chrom/Contig</span>:\
      <select class="txtSelect" id="RC-selectChrom" style="width:66px"></select>&nbsp;<span id="RC-numChroms"></span></td>\
    <td class="nwrap"><span class="flyhelp" id="RC-filterChromRangeLabel">Range</span>:\
      <input type="text" class="numSearch" id="RC-chromRange" value="" size=24></td>\
    <td class="nwrap">\
      <input type="button" id="RC-OpenIGV" value="View in IGV" style="width:76px"></td>\
  </tr></table>\
</div>\
<div id="RC-tooltip" style="display:none">\
  <div><span id="RC-tooltip-close" title="Close" class="help-box ui-icon ui-icon-close"></span></div>\
  <div><span id="RC-tooltip-zoomout" title="Zoom out from this region" class="help-box ui-icon ui-icon-zoomout"></span></div>\
  <div><span id="RC-tooltip-center" title="Center view on this region" class="help-box ui-icon ui-icon-arrowthick-2-e-w"></span></div>\
  <div><span id="RC-tooltip-zoomin" title="Zoom in on this region" class="help-box ui-icon ui-icon-zoomin"></span></div>\
  <div id="RC-tooltip-body"></div>\
</div>\
<div id="RC-helptext" class="helpblock" style="z-index:10;display:none">\
This chart shows the base coverage due to reads aligned across the whole reference.<br/><br/>\
For a reference of multiple contigs (chromosomes), the initial view will show total<br/>\
coverage as a data bar per contig. If there are many contigs each bar itself may<br/>\
represent a binned average over a smaller number of contigs. Click-and-drag or double-<br/>\
click the mouse pointer to zoom in on either a range of contigs or bases of a single contig.<br/><br/>\
With a single contig (chromosome) in view each data bar (or overlay point) will represent<br/>\
the average base coveage over a range of the reference genome. Click-and-drag or double-<br/>\
click to zoom in to regions along the contig, or use the Range field of the options panel.<br/><br/>\
Double-click in the white-space around the plotted data to zoom out (by upto 10x).<br/>\
Click on the Zoom Out button to return to the coverage view across the whole contig<br/>\
currently in view or the coverage-per-contig view for the whole reference.<br/><br/>\
Hover the mouse pointer over a data bar (or overlay point) to review minimal information<br/>\
or click on the data to show a detailed information box that remains until dismissed.<br/><br/>\
More display features are available using the Plot and Overlay selectors in the title bar<br/>\
or accessed using the view options panel (by clicking on the adjacent spy-glass icon).<br/>\
Depending on the total number of contigs, the "Zoom In" button may also be available.<br/>\
Look for additional tool-tip help on or near the controls provided.\
</div>\
<div id="RC-mask" class="grid-mask"></div>\
<div id="RC-dialog" class="tools-dialog" style="width:450px;display:none">\
  <div id="RC-dialog-title" class="title">Export Targets in View</div>\
  <div id="RC-dialog-content" class="content">...</div>\
  <div id="RC-dialog-buttons" class="buttons">\
    <input type="button" value="OK" id="RC-exportOK">\
    <input type="button" value="Cancel" onclick="$(\'#RC-dialog\').hide();$(\'#RC-mask\').hide();">\
  </div>\
</div>\
<input type="hidden" id="RC-ViewRequest"/>\
');

$(function () {

  // check placer element exists
  if( !$('#ReferenceCoverageChart').length ) return;

  // check browser environment
  var fixIE = (typeof G_vmlCanvasManager != 'undefined');
  var useFlash = (typeof FlashCanvas != 'undefined');
  var useExCan = (fixIE && !useFlash);

  // minimum sizes for chart widget
  var def_minWidth = 625;
  var def_minHeight = 200;

  // configure widget size and file used from placement div attributes
  var bbcFile = $("#ReferenceCoverageChart").attr("bbcfile");
  if( bbcFile == undefined  || bbcFile == "" ) {
    //alert("ERROR on page: ReferenceCoverageChart widget requires attribute 'bbcfile' is set.");
    $('#ReferenceCoverageChart').hide();
    return;
  }
  var chrcovFile = $("#ReferenceCoverageChart").attr("chrcovfile");
  if( chrcovFile == undefined || chrcovFile == "" ) {
    //alert("ERROR on page: ReferenceCoverageChart widget requires attribute 'chrcovfile' is set.");
    $('#ReferenceCoverageChart').hide();
    return;
  }
  var cbcsize = 1000; // coarse binning size
  var annoFile = $("#ReferenceCoverageChart").attr("annofile");
  if( annoFile == undefined ) annoFile = '';
  var wgncovFile = $("#ReferenceCoverageChart").attr("wgncovfile");
  if( wgncovFile == undefined ) wgncovFile = '';

  var wholeGenome = $("#ReferenceCoverageChart").attr("genome");
  wholeGenome = (wholeGenome != undefined);

  var startCollapsed = $("#ReferenceCoverageChart").attr("collapse");
  startCollapsed = (startCollapsed != undefined);

  var startShowOptions = $("#ReferenceCoverageChart").attr("showoptions");
  startShowOptions = (startShowOptions != undefined);

  var startOutlierOffScale = $("#ReferenceCoverageChart").attr("outlieroffscale");
  startOutlierOffScale = (startOutlierOffScale != undefined);

  var nooverlay = $("#ReferenceCoverageChart").attr("nooverlay");
  nooverlay = (nooverlay != undefined);
  if( nooverlay ) {
    $('#RC-overlayLabel').hide();
    $('#RC-overPlot').hide();
  }

  var tmp = $('#ReferenceCoverageChart').width();
  if( tmp < def_minWidth ) tmp = def_minWidth;
  $("#RC-chart").width(tmp);
  tmp = $('#ReferenceCoverageChart').height();
  if( tmp < def_minHeight ) tmp = def_minHeight;
  $("#RC-chart").height(tmp);
  $("#RC-placeholder").height(tmp-36);
  $("#ReferenceCoverageChart").css('height','auto');

  $("#RC-controlpanel").appendTo('#RC-titlebar');
  $("#RC-chart").appendTo('#ReferenceCoverageChart');
  $("#RC-chart").show();
  $('#RC-chart').css("height","auto");

  // some default values for plot display
  var def_minPoints = 11;  // odd so one is in the middle
  var def_numPoints = 200;
  var def_tinyValue = 0.00001;
  var def_outlierFactor = 4;
  var disableTitleBar = false;
  var placeholder = $("#RC-placeholder");
  var timeout = null;

  var dblclickUnzoomFac = 10;

  var resiz_def = {
    alsoResize: "#RC-placeholder",
    minWidth:def_minWidth,
    minHeight:def_minHeight,
    handles:"e,s,se",
    resize:function(e,u){ updatePlot(); }
  };
  $('#RC-chart').resizable(resiz_def);

  function disableShowTargets() {
    $('#RC-showTargetsOption').hide();
    $('#RC-showTargets').prop('checked',false);
    plotParams.showTargets = false;
    baseRangeParam.annofile = '';
  }

  placeholder.bind("mouseleave", function() {
    if( !lastHoverBar.sticky ) hideTooltip();
  });

  $("#RC-slider").slider({min:0,step:1});
  var sliderHandle = $('.ui-slider.grid-microslider .ui-slider-handle');

  $("#RC-collapsePlot").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#RC-plotspace').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#RC-controlpanel').slideUp();
      $('.RC-shy').fadeOut(400);
      $('#RC-chart').resizable('destroy');
      $('#RC-noncanvas').slideUp('slow');
      hideTooltip();
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('.RC-shy').fadeIn(400);
      $('#RC-noncanvas').slideDown('slow',function(){
        $('#RC-chart').resizable(resiz_def);
      });
    }
    if( nooverlay ) {
      $('#RC-overlayLabel').hide();
      $('#RC-overPlot').hide();
    }
    $("#RC-chart").css('height','auto');
  });

  $("#RC-toggleControls").click(function(e) {
    if( disableTitleBar ) return;
    $('#RC-chart').css("height","auto");
    if( $('#RC-controlpanel').is(":visible") ) {
      $('#RC-controlpanel').slideUp();
    } else {
      $('#RC-controlpanel').slideDown();
    }
  });

  $("#RC-help").click( function() {
    var offset = $("#RC-help").offset();
    var ypos = offset.left - $('#RC-helptext').width();
    $("#RC-help").removeAttr("title");
    $('#RC-helptext').css({
      position: 'absolute', display: 'none',
      top: offset.top+16, left: ypos+8
    }).appendTo("body").slideDown();
  });

  $("#RC-help").hover( null, function() {
    $('#RC-helptext').fadeOut(200);
    $("#RC-help").attr( "title", "Click for help." );
  });

  //$('#RC-chart').noContext();

  function rightClickMenu(e) {
    alert("r-click");
  }

  // attempt to disable defalt context menu and enable r-click
  if( useFlash ) {
    // only works for flashcanvas pro!
    FlashCanvas.setOptions( {disableContextMenu : true} );
    // instead use area outside of canvas
    //$('#RC-noncanvas').noContext();
    //$('#RC-noncanvas').rightClick(rightClickMenu);
  } else {
    //$('#RC-chart').noContext();
    //$('#RC-chart').rightClick(rightClickMenu);
  }

  function customizeChart() {
    // add fly-over help to controls here in case need to customize for chart data
    $("#RC-plotLabel").attr( "title",
      "Select the how the data is plotted.\n'Total Reads' shows bar plots of base reads aligned "+
      "to both DNA strands, whereas 'Strand Reads' plots the numbers of forward and reverse (DNA strand) base reads "+
      "separately, above and below the 0 reads line. If a set of target regions was specified, the numbers of reads "+
      "that were inside or outside of these regions are shown by stacked colors of each data bar." );
    $("#RC-overlayLabel").attr( "title",
      "Select a particular property of the data to plot as an overlay of points parallel to each data bar. " +
      "For example, the Strand Bias can be overlayed with the Total Reads plot to see if there is any obvious strand "+
      "bias for the reference regions in view. Whether there is specific or general strand bias "+
      "may depend on the current zoom level to those regions (i.e. the region size represented by the binned data). "+
      "Adding an overlay plot is also useful for identifying very low coverage regions, since these points are not "+
      "plotted for any region with 0 base coverage." );
    $("#RC-autoZoomLabel").attr( "title",
      "Select to allow the current view to automatically scale the Base Read Depth (y-axis) to the largest peak within view. "+
      "The default (unchecked) behavior is to automatically scale to the largest peak within the current slider range." ); 
    $("#RC-offScaleOutlierLabel").attr( "title",
      "Select to indicate that a single outlier data point is plotted off-scale. "+
      "The y-axis is re-scaled so that the over-represented contig (or range) does not hide the relative representation "+
      "of other data curently in view. This option only becomes available when the maximum value (height) of any "+
      "data point (bar) is at least "+def_outlierFactor+" times greater than all others in view." );
    $("#RC-showLegendLabel").attr( "title", "Select whether the legend is displayed over the plot area." );
    $("#RC-showTargetsLabel").attr( "title", "Select whether the target coverage bar is displayed over the plots. "+
      "Clicking on one of these bars may be used to bring a single target into view in the Amplicon/Target Coverage Chart. "+
      "Target coverage bars do not appear in the whole chromomsome/contig view (at maximum zoom out)." );
    $("#RC-numPointsLabel").attr( "title",
      "This value specifies the maximum number of data bars and overlay points to display. Typically each bar (or point) "+
      "plotted will represent the binned totals and averages of many individual base regions along the genome. If there is "+
      "less data to plot than this value, e.g. when in the maximum zoom-out mode showing coverage per chromosome, the "+
      "number of bars actually plotted is displayed in parentheses. This value may be set to any value 10 and 1000, "+
      "although values greater than 200 are not recommended as this may make selection of individual bars difficult.\n"+
      "Note that the number of bars represented in the 'Zoom In' view is fixed at 200." );
    $("#RC-export").attr( "title", "Click to open Export Reference Coverage in View dialog." );
    $("#RC-dialog-title").html( "Export Reference Coverage in View" );
    $("#RC-filterChromLabel").attr( "title",
      "Use this selector to select a particular chromosome (or contig) of the reference to view or to go to an overview " +
      "across the whole reference by selecting the 'ALL' value. " +
      "You may also change to a Chrom/Contig selection by typing its full name (or ID) in the Range field." );
    $("#RC-filterChromRangeLabel").attr( "title",
      "Edit current contig/chromosome range in view using the format <contig>:<start>-<end>. "+
      "The <end> coordinate may be omitted to center the view on the <start>, or just the <contig> typed to "+
      "view the whole contig. If <contig> is omitted the contig already in view is assumed. The range may be "+
      "modified to fit the contig, limits of the binned data or meet the number of Bars/Points specified. "+
      "Large ranges may be adjusted to the nearest multiple of 1,000 to make use of pre-binned averaged coverage." );
    $("#RC-OpenIGV").attr( "title",
      "Click this button to open an instance of IGV (Integrated Genome Viewer) with whatever Chrom/Contig and Range selection "+
      "is currently in view. Your target/amplicon regions are also uploaded as a separate annotation track if a target regions "+
      "file was specified." );
    $("#RC-help").attr( "title", "Click for help." );
  }

  var plotStats = {
    xTitle : "Reference Range",
    defNumPoints : def_numPoints,
    minNumPoints : def_minPoints,
    maxNumPoints : 1000,
    maxSideBars : 500,
    maxXaxisLabels : 30,
    multiChrom : false,
    zoomInOption : false,
    zoomInActive : false,
    binnedChroms : false,
    zoomChrom: false,
    chromSrtNum : 1,
    chromsInView : 0,
    basesInView : 0,
    baseBinSize : 0,
    binnedBases : false,
    onTargets : false,
    numFields : 0,
    numPlots : 0,
    numPoints : 0,
    minX : 0,
    maxX : 0,
    minY : 0,
    maxY : 0,
    tooltipZero : 0,
    totalChroms : 0,
    chromLength : 0,
    chrList : "",
    chrLens : {},
    chrIdx : {},
    chromLbins : 0,
    chromRbins : 0,
    baseLbins : 0,
    baseRbins : 0,
    sliderMotive : false,
    sliderScale : 0,
    sliderRfPos : 0,
    sliderShift : 0
  };

  var plotParams = {
    resetYScale: false,
    showLegend : true,
    showTargets : true,
    autoZoom : false,
    offScaleOutlier : false,
    numPoints : def_numPoints,
    aveBase : 1,
    barAxis : 0,
    overPlot : 0,
    zoomMode : 1,
    dblCenter : false
  };

  var baseRangeParam = {
    outfile : '',
    options : '',
    bbcfile : bbcFile,
    annofile : annoFile,
    annonumflds : 1,
    annofields : '3',
    annotitles : 'targets',
    chrom : '',
    pos_srt : 0,
    pos_end : 0,
    maxrows : 200,
    srt_bin : 0,
    end_bin : 0,
    clipleft : 0,
    clipright : 100
  };

  var i = bbcFile.lastIndexOf('/');
  var fpath = i < 0 ? "" : bbcFile.substring(0,i+1);

  var contigRangeParam = {
    filename : fpath+chrcovFile,
    outfile : '',
    startline : 1,
    numlines : 1,
    binsize : 1,
    binsrt : 0,
    binend : 0,
    bedcoords : 0,
    headlines : 1,
    startfield : 1,
    numfields : 7
  };

  var DataField = {
    contig_id : 0,
    pos_start : 1,
    pos_end : 2,
    fwd_reads : 3,
    rev_reads : 4,
    fwd_ont : 5,
    rev_ont : 6
  };

  var LegendLabels = {
    allReads : "Total Reads",
    fwdReads : "Forward Reads",
    revReads : "Reverse Reads",
    offReads : "Off-target",
    ontReads : "On-target",
    fwdOffReads : "Forward Off-target",
    fwdOntReads : "Forward On-target",
    revOffReads : "Reverse Off-target",
    revOntReads : "Reverse On-target",
    percentOntarg : "Percent On-Target",
    fwdBias : "Fwd Strand Bias"
  }

  var ColorSet = {
    allReads : "rgb(128,160,192)",
    fwdReads : "rgb(240,120,100)",
    revReads : "rgb(100,240,120)",
    offReads : "rgb(128,160,192)",
    ontReads : "rgb(0,0,128)",
    fwdOffReads : "rgb(240,120,100)",
    revOffReads : "rgb(100,240,120)",
    fwdOntReads: "rgb(160,32,32)",
    revOntReads : "rgb(32,160,32)",
    percentOntarg : "rgb(180,143,32)",
    fwdBias : "rgb(220,96,200)",
    percentGC : "rgb(255,0,128)",
    targLen : "rgb(32,96,255)",
    gcBias : "rgb(190,190,190)"
  }

  function setUnzoomTitle(zoomin) {
    var txt;
    if( zoomin ) {
      txt = "Zoom in to a special view of coverage over the reference as a single sequence with all chromsomes\n"+
      "(or contigs) laid end-to-end. This view is only available for references with 2 to 30 contigs.\n"+
      "Coverage across each is distributed over a number of bins that is apportioned by relative sizes of\n"+
      "the chromsomes, with each chromsome represented by at least one bin.";
    } else {
      txt = "Zoom out to the maximum stop distance for the current view: Coverage over the whole chromosome\n"+
      "or to the initial view for coverage off all chomosomes/contigs in the whole reference.";
    }
    plotStats.zoomInOption = zoomin;
    $("#RC-unzoomToggle").val(zoomin ? "Zoom In" : "Zoom Out");
    $("#RC-unzoomToggle").attr( "title", txt );
  }

  function checkZoomInOption() {
    if( wgncovFile === '' ) return;
    if( plotStats.zoomInOption ) {
      setUnzoomTitle(false);
    } else if( plotStats.multiChrom && plotStats.totalChroms == plotStats.chromsInView ) {
      if( !plotStats.zoomInActive ) setUnzoomTitle(true);
    }
  }

  // Sets up range loading in a multi-contig views and reloads data if necessary - return true if ziew updated
  // If in Zoom In mode either switch a standard single contig zoom or just set the number of contigs in view.
  function setLoadChromRange(binSrt,binEnd) {
    var srtChr = dataTable[binSrt][DataField.contig_id];
    var endChr = dataTable[binEnd][DataField.contig_id];
    var i = srtChr.indexOf(' - ');
    if( i >= 0 ) srtChr = srtChr.substr(0,i);
    i = endChr.indexOf(' - ');
    if( i >= 0 ) endChr = endChr.substr(i+3);
    plotStats.chromSrtNum = plotStats.chrIdx[srtChr];
    plotStats.chromsInView = plotStats.chrIdx[endChr] - plotStats.chromSrtNum + 1;
    if( srtChr == endChr ) {
      zoomDataRange(srtChr,dataTable[binSrt][DataField.pos_start],dataTable[binEnd][DataField.pos_end]);
      return true;
    }
    if( plotStats.zoomInActive ) return false;
    if( plotStats.chromsInView < plotStats.minNumPoints ) {
      plotStats.chromSrtNum -= (plotStats.minNumPoints-plotStats.chromsInView) / 2;
      if( plotStats.chromSrtNum < 1 ) plotStats.chromSrtNum = 1;
      plotStats.chromsInView = plotStats.minNumPoints;
    }
    // expand range for over-loading for slider
    var numPoints = plotStats.maxX - plotStats.minX;
    var binsize = plotStats.chromsInView / numPoints;
    if( binsize < 1 ) binsize = 1;
    if( numPoints > plotStats.maxSideBars ) numPoints = plotStats.maxSideBars;
    var lbins = (plotStats.chromSrtNum-1)/binsize;
    if( lbins > numPoints ) lbins = numPoints;
    var rbins = (plotStats.totalChroms-plotStats.chromSrtNum-plotStats.chromsInView)/binsize;
    if( rbins > numPoints ) rbins = numPoints;
    contigRangeParam.binsize = binsize;
    plotStats.chromLbins = lbins;
    plotStats.chromRbins = rbins;
    if( plotStats.binnedChroms ) {
      zoomData();
      return true;
    }
    return false;
  }

  // return true if bin is (now) at center
  function centerViewOnBin(binNum) {
    if( plotStats.numPlots <= 0 || binNum < 0 ) return false;
    // binNum 0-based so 99 (bin#100) is center of 200 bins and 100 (bin#101) is center of 201 bins
    var shift = binNum - ((plotStats.minX+plotStats.maxX-1) >> 1);
    if( shift == 0 ) return true;
    zoomViewToBinRange( plotStats.minX+shift, plotStats.maxX+shift-1 );
    return true;
  }

  // (re)initiailize page from default user options
  customizeChart();
  updateGUIPlotParams();

  // --------- initialize plot bindings - controls inside plot area ----------

  var fieldIds = [];
  var dataTable = [];
  var plotData = [];
  var options = [];
  var pointMap = [];
  var plotObj = null;
  var canvas = null;

  var lastHoverBar = {binNum:-1, isRev:false, clickItem:null, postZoom:false, sticky:false, label:'' };

  // plotselected (area) is mapped to zoom - which is quite complicated for this app.
  placeholder.bind("plotselected", function(event, ranges) {
    if( plotParams.zoomMode == 2 ) {
      plotStats.minY = options.yaxes[0].min = ranges.yaxis.from;
      plotStats.maxY = options.yaxes[0].max = ranges.yaxis.to;
    }
    lastHoverBar.postZoom = true;  // prevent help pop-up
    zoomViewToBinRange( Math.floor(ranges.xaxis.from), Math.floor(ranges.xaxis.to) );
  });

  placeholder.bind("plothover", function(event, pos, item) {
    var hoverTip = !lastHoverBar.sticky;
    if( cursorOverPlot(pos.x,pos.y) ) {
      if( cursorOverItem(pos,item) ) {
        setCursor('pointer');
        if( hoverTip ) showTooltip(item,pos,false);
      } else {
        setCursor('crosshair');
        if( hoverTip ) hideTooltip();
      }
    } else {
      setCursor('default');
      if( hoverTip ) hideTooltip();
    }
  });

  var numClicks = 0;
  var clickTimer = null;
  placeholder.bind("plotclick", function(e,pos,item) {
    // manual implement of dblclick since single click always fires
    if( ++numClicks > 1 ) {
      clearTimeout(clickTimer);
      numClicks = 0;
      var binNum = item ? Math.floor(pos.x) : -1;
      if( zoomViewOnBin( binNum, true ) ) hideTooltip();
      return;
    }
    // ignore false triggering due to mouse selection for zoom
    if( lastHoverBar.postZoom ) {
      lastHoverBar.postZoom = false;
      numClicks = 0;
      return;
    }
    // defer click event to enable dblclick catch
    clickTimer = setTimeout( function() {
      if( cursorOverItem(pos,item) ) {
        showTooltip(item,pos,true);
        lastHoverBar.clickItem = item;
        if( item ) plotObj.highlight(item.series,item.datapoint);
      } else {
        hideTooltip();
      }
      numClicks = 0;
    }, 250 );
  });

  placeholder.bind("mouseleave", function() {
    setCursor('default');
  });

  function setCursor(curs) {
    if( curs == null || curs == "" )
       curs = 'default';
    if( useFlash && canvas != null ) {
      // prevents tooltip from appearing!
      //FlashCanvas.setCursor(canvas,curs);
    } else {
      document.body.style.cursor = curs;
    }
  }

  function cursorOverPlot(x,y) {
    return plotStats.numPlots > 0 && x >= plotStats.minX && x < plotStats.maxX && y >= plotStats.minY && y <= plotStats.maxY;
  }

  function cursorOverItem(pos,item) {
    if( pos.x >= plotStats.numPoints ) return false;
    return item || Math.abs(pos.y) < plotStats.tooltipZero;
  }

  function hideTooltip() {
    if( plotObj ) plotObj.unhighlight();
    $("#RC-tooltip").hide();
    lastHoverBar.binNum = -1;
    lastHoverBar.clickItem = null;
    lastHoverBar.sticky = false;
    lastHoverBar.label = '';
  }

  function showTooltip(item,pos,sticky) {
    if( timeout != null ) {
      clearTimeout(timeout);
      timeout = null;
    }
    // item not passed for 0-width bars: if passed check for overlay points
    var isRev = pos.y < 0;
    var trgbar = pos.trgbar ? true : false;
    var label = '', bgColor;
    if( item ) {
      var tmp = item.series.label;
      if( tmp == LegendLabels.fwdBias || tmp == LegendLabels.percentOntarg ) label = tmp;
    }
    if( trgbar ) {
      bgColor = "#FF8";
    } else if( label != '' ) {
      bgColor = item.series.color;
    } else if( plotParams.barAxis == 0 ) {
      label = LegendLabels.allReads;
      bgColor = ColorSet.allReads;
    } else {
      label = isRev ? LegendLabels.revReads : LegendLabels.fwdReads;
      bgColor = isRev ? ColorSet.revReads : ColorSet.fwdReads;
    }
    // at hi-res can trigger for bin just outside view!
    var binNum = Math.floor(pos.x);
    if( binNum < plotStats.minX ) binNum = plotStats.minX;
    if( binNum >= plotStats.maxX ) binNum = plotStats.maxX-1;
    if( !trgbar && lastHoverBar.binNum == binNum && lastHoverBar.sticky == sticky &&
      lastHoverBar.isRev == isRev && lastHoverBar.label == label ) return;
    hideTooltip();
    // correct for over-approximate bin selection for point hover with missing data points
    var clickBar = trgbar || dataBar(label);
    if( !clickBar ) {
      // if item is available try to map points throu
      if( item != null && pointMap.length > 0 ) binNum = pointMap[item.dataIndex];
      if( dataTable[binNum][DataField.fwd_reads]+dataTable[binNum][DataField.rev_reads] == 0 ) return;
    }
    lastHoverBar.binNum = binNum;
    lastHoverBar.isRev = isRev;
    lastHoverBar.sticky = sticky;
    lastHoverBar.label = label;
    if( trgbar ) {
      $('#RC-tooltip-body').html( resolveTargetBar(binNum) );
    } else if( sticky ) {
      $('#RC-tooltip-body').html( tooltipMessage(label,binNum) );
    } else {
      $('#RC-tooltip-body').html( tooltipHint(label,binNum) );
    }
    var whiteText = (label === LegendLabels.fwdBias || label === LegendLabels.percentOntarg);
    var posx = pos.pageX+10;
    var posy = pos.pageY-10;
    var minTipWidth = 0;
    if( sticky ) {
      if( trgbar ) {
        minTipWidth = 240;
      } else if( clickBar ) {
        minTipWidth = plotParams.barAxis ? 230 : 210;
      }
      var cof = $('#RC-chart').offset();
      var ht = $('#RC-tooltip').height();
      var ymax = cof.top + $('#RC-chart').height() - ht;
      posy = pos.pageY - $('#RC-tooltip').height()/2;
      if( posy > ymax ) posy = ymax;
      if( posy < cof.top-4 ) posy = cof.top-4;
      var xmid = cof.left + $('#RC-chart').width()/2;
      if( pos.pageX > xmid ) posx = pos.pageX - $('#RC-tooltip').width() - 26;
    }
    $('#RC-tooltip').css({
      position: 'absolute', left: posx, top: posy, minWidth: minTipWidth,
      background: bgColor, padding: '3px '+(sticky ? '7px' : '4px'),
      color: whiteText ? "white" : "black",
      border: (sticky ? 2 : 1)+'px solid #444',
      opacity: (sticky ? 1: 0.7), 'z-index': 20
    }).appendTo("body").fadeIn(sticky ? 10 : 100);
    if( !sticky ) {
      timeout = setTimeout( function() { hideTooltip(); }, 200 );
    }
  }

  function resolveTargetBar(bin) {
    $('#RC-tooltip-close').show();
    var zoomOut = !noZoomOut();
    $('#RC-tooltip-zoomout').toggle(zoomOut);
    $('#RC-tooltip-center').toggle(zoomOut);
    $('#RC-tooltip-zoomin').toggle( plotStats.baseBinSize > 1 );
    var trg = dataTable[bin][7];
    var targets = trg.split(",");
    var br = "<br/>";
    var msg = "Resolve&nbsp;Multiple&nbsp;Targets:"+br;
    msg += "<div style='text-align:center'>";
    for( var i = 0; i < targets.length; ++i ) {
       var tx = targets[i];
       var ti = tx.startsWith('...(') ? "Zoom in on region" : "View target in Amplicon/Target Coverage Chart";
       msg += "<a bin='"+bin+"' class='RC-targetResolve' title='"+ti+
         "'style='color:black;text-decoration:underline;cursor:pointer'>"+tx+"</a>"+br;
    }
    return msg+"</div>";
  }

  function targetBarClick(e) {
    var bin = $(this).attr("bin");
    var trg = dataTable[bin][7];
    if( trg.indexOf(',') < 0 ) {
      $("#TC-ViewRequest").val(trg).change();
    } else {
      showTooltip( null, { 'x':bin,'y':0,'pageX':e.pageX,'pageY':e.pageY,'trgbar':true }, true );
    }
  }

  // bind the on-click event to body once for later use with class
  $('body').on( 'click', 'a.RC-targetResolve', function() {
    var bin = $(this).attr("bin");
    var trg = this.text;
    hideTooltip();
    if( trg.startsWith('...(') ) {
      zoomViewOnBin( bin, true );
    } else {
      $("#TC-ViewRequest").val(trg).change();
    }
  });

  $('#RC-tooltip-close').click( function() {
    hideTooltip();
  });

  $('#RC-tooltip-zoomout').click( function() {
    if( zoomViewOnBin( lastHoverBar.binNum, false ) ) hideTooltip();
  });

  $('#RC-tooltip-center').click( function() {
    if( centerViewOnBin( lastHoverBar.binNum ) ) hideTooltip();
  });

  $('#RC-tooltip-zoomin').click( function() {
    if( zoomViewOnBin( lastHoverBar.binNum, true ) ) hideTooltip();
  });

  function dataBar(id) {
    return (id === LegendLabels.fwdReads || id === LegendLabels.revReads || id === LegendLabels.allReads);
  }

  function tooltipHint(id,bin) {
    $('#RC-tooltip-close').hide();
    $('#RC-tooltip-zoomout').hide();
    $('#RC-tooltip-center').hide();
    $('#RC-tooltip-zoomin').hide();
    if( dataBar(id) ) {
      var chr = dataTable[bin][DataField.contig_id];
      if( plotStats.binnedChroms ) {
        return chr;
      } else if( dataTable[bin][DataField.pos_start] == 1 && dataTable[bin][DataField.pos_end] == plotStats.chrLens[chr] ) {
        return chr;
      } else if( dataTable[bin][DataField.pos_start] == dataTable[bin][DataField.pos_end] ) {
        return chr+":"+commify(dataTable[bin][DataField.pos_start]);
      }
      return chr+":"+commify(dataTable[bin][DataField.pos_start])+"-"+commify(dataTable[bin][DataField.pos_end]);
    }
    var totalReads = dataTable[bin][DataField.fwd_reads] + dataTable[bin][DataField.rev_reads];
    if( id === LegendLabels.percentOntarg ) {
      var onTarget = dataTable[bin][DataField.fwd_ont]+dataTable[bin][DataField.rev_ont];
      return (totalReads > 0 ? sigfig(100 * onTarget / totalReads) : 0)+'%';
    } else if( id === LegendLabels.fwdBias ) {
      return sigfig(totalReads > 0 ? 100 * dataTable[bin][DataField.fwd_reads] / totalReads : 50)+'%';
    }
    return '?';
  }

  function tooltipMessage(id,bin) {
    $('#RC-tooltip-close').show();
    var zoomOut = !noZoomOut();
    $('#RC-tooltip-zoomout').toggle(zoomOut);
    $('#RC-tooltip-center').toggle(zoomOut);
    $('#RC-tooltip-zoomin').toggle( plotStats.baseBinSize > 1 );
    var br = "<br/>";
    var i = id.indexOf(' ');
    var dirStr = id.substr(0,i+1);
    var dir = dirStr.charAt(0);
    var regionLen = dataTable[bin][DataField.pos_end];
    var numReads = dataTable[bin][DataField.fwd_reads]+dataTable[bin][DataField.rev_reads];
    var msg = "(Bin#"+(bin+1-plotStats.minX)+")"+br;
    if( plotStats.binnedChroms ) {
      msg += "Contigs: "+dataTable[bin][DataField.contig_id]+br;
      msg += "Number of contigs: "+dataTable[bin][DataField.pos_start]+br;
      msg += "Total contig length: "+commify(regionLen)+br;
    } else {
      regionLen += 1-dataTable[bin][DataField.pos_start];
      msg += "Contig: "+dataTable[bin][DataField.contig_id]+br;
      msg += "Region: "+commify(dataTable[bin][DataField.pos_start])+(regionLen>1 ? "-"+commify(dataTable[bin][DataField.pos_end]) : "")+br;
      msg += "Region length: "+commify(regionLen)+br;
    }
    if( id == LegendLabels.fwdBias ) {
      var bias = numReads >  0 ? 100 * dataTable[bin][DataField.fwd_reads] / numReads : 50;
      if( numReads >  0 ) msg += "Forward reads: "+sigfig(bias)+'%'+br;
      msg += bias >= 50 ? "Forward" : "Reverse";
      bias = Math.abs(2*bias-100);
      return msg + " bias: "+sigfig(bias)+'%'+br;
    } else if( id == LegendLabels.percentOntarg ) {
      msg += "Total base reads: "+commify(numReads)+br;
      var onTarget = dataTable[bin][DataField.fwd_ont]+dataTable[bin][DataField.rev_ont];
      var pcOntarg = numReads >  0 ? 100 * onTarget / numReads : 0;
      if( numReads >  0 ) msg += "On-target reads: "+sigfig(pcOntarg)+'%'+br;
      return msg;
    }
    var dirReads = dir == 'T' ? numReads : dataTable[bin][dir == 'F' ? DataField.fwd_reads : DataField.rev_reads];
    var aveReads = dirReads / (regionLen);
    msg += dirStr + "base reads: "+commify(dirReads)+br;
    msg += "Average "+(dir == 'T' ? "" : dirStr.toLowerCase())+"base read depth:  "+commify(sigfig(aveReads))+br;
    if( dir == 'T' ) {
      var bias = numReads >  0 ? 100 * dataTable[bin][DataField.fwd_reads] / numReads : 0;
      msg += "Percent forward reads: "+sigfig(bias)+'%'+br;
    } else {
      var bias = numReads >  0 ? 100 * dirReads / numReads : 0;
      msg += "Percent "+(dir == 'F' ? "forward" : "reverse")+" reads: "+sigfig(bias)+'%'+br;
    }
    if( plotStats.onTargets ) {
      var onTarget = dataTable[bin][DataField.fwd_ont]+dataTable[bin][DataField.rev_ont];
      var dirTarg = dir == 'T' ? onTarget : dataTable[bin][dir == 'F' ? DataField.fwd_ont : DataField.rev_ont];
      var pcOntarg = dirReads >  0 ? 100 * dirTarg / dirReads : 0;
      msg += dirStr + "on-target base reads: "+commify(dirTarg)+br;
      msg += dirStr + "off-target base reads: "+commify(dirReads-dirTarg)+br;
      msg += dirStr + "on-target fraction: "+sigfig(pcOntarg)+'%'+br;
    }
    return msg;
  }

  function commify(val) {
    var jrs = val.toString();
    var dps = "";
    var i = jrs.indexOf('.');
    if( i >= 0 ) {
      dps = jrs.substring(i);
      jrs = jrs.substring(0,i);
    }
    return jrs.replace(/(\d)(?=(\d\d\d)+(?!\d))/g, "$1,")+dps;
  }

  function roundAxis( maxVal ) {
    if( maxVal == 0 ) return 0;
    var sgn = maxVal < 0 ? -1 : 1;
    maxVal = Math.abs(maxVal);
    var b = Math.pow( 10, Math.round(Math.log(maxVal)/Math.LN10)-1 );
    return sgn * b * Math.floor(1+maxVal/b);
  }

  function percentFormat(val, axis) {
    return ''+val.toFixed(axis.tickDecimals)+'%';
  }

  function absFormat(val, axis) {
    return ''+Math.abs(val.toFixed(axis.tickDecimals));
  }

  function sigfig(val) {
    var av = Math.abs(val);
    if( av == parseInt(av) ) return val.toFixed(0);
    if( av >= 100 ) return val.toFixed(0);
    if( av >= 10 ) return val.toFixed(1);
    if( av >= 1 ) return val.toFixed(2);
    if( av >= 0.1 ) return val.toFixed(3);
    return val.toFixed(3);
  }

  // --------- Set up charting controls outside of plot area ---------

  $('#snapShot').click(function() {
    if( canvas != null ) {
      canvas2png(canvas);
    }
  });

  function updateGUIPlotParams() {
    $('.RC-selectParam#barAxis').val(plotParams.barAxis);
    $('.RC-selectParam#overPlot').val(plotParams.overPlot);
    $('#RC-numPoints').val(plotParams.numPoints);
    $('#RC-autoZoom').attr('checked',plotParams.offScaleOutlier);
    $('#RC-offScaleOutlier').attr('checked',plotParams.offScaleOutlier);
    $('#RC-showLegend').attr('checked',plotParams.showLegend);
    $('#RC-showTargets').attr('checked',plotParams.showTargets);
    $('#RC-selectChrom').val(baseRangeParam.chrom);
  }

  $('#RC-autoZoom').change(function() {
    plotParams.autoZoom = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $('#RC-offScaleOutlier').change(function() {
    plotParams.offScaleOutlier = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $('#RC-showLegend').change(function() {
    plotParams.showLegend = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $('#RC-showTargets').change(function() {
    plotParams.showTargets = ($(this).attr("checked") == "checked");
    // show targets not implemented for whole contig view (stil using public file for view)
    if( baseRangeParam.chrom === '' && !plotStats.zoomInActive ) return;
    if( plotParams.showTargets ) {
      if( plotStats.zoomInActive ) {
        updatePlot();
      } else {
        // reload data to get anotation - last load range is dynamic and has to be recalculated
        baseRangeParam.annofile = annoFile;
        setContigRange( $('#RC-chromRange').val() );
        zoomData();
      }
    } else {
      baseRangeParam.annofile = '';
      updatePlot();
    }
  });

  $('#RC-numPoints').change(function() {
    var val = this.value.trim();
    val = (val == '') ? plotStats.defNumPoints : Math.floor(val);
    if( isNaN(val) ) {
      val = plotParams.numPoints;
    } else if( val < plotStats.minNumPoints ) {
      val = plotStats.minNumPoints;
    } else if( val > plotStats.maxNumPoints ) {
      val = plotStats.maxNumPoints;
    }
    this.value = val;
    if( val != plotParams.numPoints ) {
      baseRangeParam.maxrows = plotParams.numPoints = val;
      // Selected bars are fixed for Zoom In mode so just update GUI
      if( plotStats.zoomInActive ) {
        if( plotParams.numPoints != plotStats.numPoints ) {
          $('#RC-numBars').text('('+plotStats.numPoints+')');
        } else {
          $('#RC-numBars').text('');
        }
      } else {
        unzoomData();
      }
    }
  });
    
  $('#RC-selectChrom').change(function() {
    updateContigView(this.value);
  });

  $('#RC-chromRange').change(function() {
    // first check for a valid (leading) chromosome ID
    var val = $.trim(this.value);
    var chr = val;
    var selChr = $('#RC-selectChrom option[value="'+chr+'"]').length;
    if( selChr > 0 ) {
      val = '';
    } else {
      var j = val.lastIndexOf(':');
      if( j >= 0 ) {
        chr = $.trim(val.substring(0,j));
        val = $.trim(val.substring(j+1));
        if( $('#RC-selectChrom option[value="'+chr+'"]').length == 0 ) chr = '';
      } else {
        chr = '';
      }
    }
    if( chr !== '' ) {
      if( val === '' ) {
        this.value = '';
        updateContigView(chr);
        return;
      }
      setContig(chr);
    } else if( baseRangeParam.chrom === '' ) {
      // range is reset if no contig selected
      this.value = '';
      return;
    }
    // parse range string and reset to valid range
    plotStats.zoomChrom = setContigRange(val);
    zoomData();
  });

  function setContigRange(rangeStr) {
    // format is 'x' or 'x-y' or 'x y' where x and y are number strings with commas removed
    // range becomes whole contig on parsing error or empty string
    // return true if a slice of the contig is selected, or false if whole contig in view
    var val = $.trim(rangeStr.replace(/,/g,''));
    var pos;
    if( val.indexOf('-') > 0 ) {
      val = val.replace(/\s/g,'');
      pos = val.split('-');
    } else {
      pos = val.split(/\s+/);
    }
    var srt = 1;
    var end = plotStats.chromLength;
    // reset view assuming already focused on contig
    baseRangeParam.maxrows = plotStats.numPoints = plotParams.numPoints;
    plotStats.minX = 0;
    plotStats.maxX = plotStats.numPoints;
    plotStats.zoomInActive = plotStats.multiChrom = false;
    // short circuit for invalid format of start location
    if( pos.length < 1 || isNaN(pos[0]) ) {
      $('#RC-chromRange').val(commify(srt)+'-'+commify(end));
      baseRangeParam.pos_srt = srt;
      baseRangeParam.pos_end = end;
      return false;
    }
    // center view around bases at single position
    if( pos.length == 1 ) {
      srt = parseInt(pos[0]);
      end = srt;
      srt -= plotStats.numPoints/2;
      srt = Math.floor(srt);
      if( srt < 1 ) srt = 1;
      end = srt + plotStats.numPoints - 1;
      if( end > plotStats.chromLength ) {
        end = plotStats.chromLength;
        srt = end - plotStats.numPoints + 1;
        if( srt < 1 ) srt = 1;
      }
    } else {
      if( pos[0] == '' || isNaN(pos[0]) ) srt = 1;
      else srt = parseInt(pos[0]);
      if( pos[1] == '' || isNaN(pos[1]) ) end = plotStats.chromLength;
      else end = parseInt(pos[1]);
      if( end < srt ) {
        var swp = end;
        end = srt;
        srt = swp;
      }
    }
    // expand range if insufficient
    if( (end-srt+1) < plotStats.numPoints ) {
      srt = Math.floor(0.5*(end+srt+1-plotStats.numPoints));
      if( srt < 1 ) srt = 1;
      end = srt + plotStats.numPoints - 1;
      if( end > plotStats.chromLength ) {
        end = plotStats.chromLength;
        srt = end - plotStats.numPoints + 1;
        if( srt < 1 ) srt = 1;
      }
    }
    $('#RC-chromRange').val(commify(srt)+'-'+commify(end));
    baseRangeParam.pos_srt = srt;
    baseRangeParam.pos_end = end;
    return (srt > 1 || end < plotStats.chromLength);
  }

  function updateContigView(chr) {
    if( chr == '' ) return;
    setContig(chr);
    unzoomData();
  }

  function setContig(chr) {
    if( chr == '' || plotStats.totalChroms < 2 ) return;
    plotStats.chromSrtNum = 1;
    if( chr == 'ALL' ) {
      plotStats.chromsInView = plotStats.totalChroms;
      plotStats.chromLength = 0;
      baseRangeParam.chrom = '';
    } else if( chr != baseRangeParam.chrom ) {
      plotStats.chromsInView = 1;
      plotStats.chromLength = plotStats.chrLens[chr];
      baseRangeParam.chrom = chr;
    } else {
      return;
    }
    $('#RC-selectChrom').val(chr);
  }

  $('.RC-selectParam').change(function() {
    plotParams[this.id] = parseInt(this.value);
    plotParams.resetYScale = this.id == 'barAxis';
    autoHideLegend();
    updatePlot();
  });

  function autoHideLegend() {
    if( plotParams.showLegend && !plotStats.onTargets && plotParams.barAxis == 0 && plotParams.overPlot == 0 ) {
      $('#RC-showLegend').attr('checked', plotParams.showLegend = false );
    } else if( !plotParams.showLegend ) {
      $('#RC-showLegend').attr('checked', plotParams.showLegend = true );
    }
  }
  
  $("#RC-unzoomToggle").click(function() {
    if( this.value == "Zoom In" ) {
      plotStats.zoomInActive = true;
    } else if( plotStats.zoomInActive ) {
      plotStats.zoomInActive = (plotStats.maxX - plotStats.minX < plotStats.numPoints);
    }
    if( !plotStats.zoomChrom ) setContig('ALL');
    unzoomData();
  });

  $('#RC-OpenIGV').click(function() {
    window.open( linkIGV( getDisplayRegion(0) ) );
  });

  function getDisplayRegion(viewBuffer) {
    var chr = $('#RC-selectChrom').val();
    if( chr == '' || chr == 'ALL' ) return '';
    var srt = baseRangeParam.pos_srt;
    var end = baseRangeParam.pos_end;
    if( srt <= 0 ) return chr;
    if( end <= 0 ) return chr+':'+end;
    // expand the region slightly for better viewing in context
    srt -= viewBuffer;
    if( srt < 1 ) srt = 1;
    end += viewBuffer;
    if( end > plotStats.chromLength ) end = plotStats.chromLength;
    return chr+':'+srt+'-'+end;
  }

  function linkIGV(region) {
    if( region == undefined || region == null || region == "" ) {
      var i = plotStats.chrList.indexOf('\t');
      if( i > 0 ) { region = plotStats.chrList.substr(0,i); }
    }
    var locpath = window.location.pathname.substring(0,window.location.pathname.lastIndexOf('/'));
    var igvURL = window.location.protocol + "//" + window.location.host + "/auth" + locpath + "/igv.php3";
    var launchURL = window.location.protocol + "//" + window.location.host + "/IgvServlet/igv";
    return launchURL + "?locus="+region+"&sessionURL="+igvURL;
  }

  // grab update request from external source (to RC)
  $('#RC-ViewRequest').change(function() {
    // ignore request if widget currently disabled
    if( disableTitleBar ) return;
    // force chart into view when request recieved - circumvents display issues if newly ploted while collapsed
    if( !$('#RC-plotspace').is(":visible") ) {
      $("#RC-collapsePlot").click();
    }
    $('#RC-chromRange').val(this.value).change();
  });

  function loadContigs() {
    var chrid = '';
    $('#RC-message').text('Loading...');
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get( chrcovFile, function(mem) {
      var lines = mem.split("\n");
      $.each(lines, function(n,row) {
        if( n > 0 ) {
          var fields = $.trim(row).split('\t');
          if( fields[0] !== '' && chrid !== fields[0] ) {
            chrid = fields[0];
            plotStats.chrList += chrid + '\t';
            plotStats.chrLens[chrid] = parseInt(fields[2]);
            plotStats.chrIdx[chrid] = n;
          }
        }
      });
    }).error(function(xhr,status,error){
      $('#RC-message').text('Failed to load from contig summary file.');
    }).success(function(){
      setChromSearch();
      $('#RC-message').text('');
    });
  }

  function setChromSearch() {
    var selObj = $('#RC-selectChrom');
    selObj.empty();
    selObj.css('width','66px');
    if( plotStats.chrList == '' ) return;
    var chrs = plotStats.chrList.split('\t');
    plotStats.totalChroms = chrs.length - 1;
    $('#RC-numChroms').text( '('+plotStats.totalChroms+')' );
    if( plotStats.totalChroms > 1 ) {
      selObj.append("<option value='ALL'>ALL</option>");
    }
    var mclen = 0;
    for( var i = 0; i < plotStats.totalChroms; ++i ) {
      selObj.append("<option value='"+chrs[i]+"'>"+chrs[i]+"</option>");
      if( chrs[i].length > mclen ) mclen = chrs[i].length;
    }
    if( mclen > 6 ) selObj.css('width','');
    // if only one chromosome then the selection cannot be changed => ensure it is the one selected
    if( plotStats.totalChroms == 1 ) {
      baseRangeParam.chrom = chrs[0];
      plotStats.chromLength = plotStats.chrLens[chrs[0]];
    }
  }

  function zoomViewOnBin(binNum,zoomIn) {
    // always perform zoom out if binNum < 0
    if( plotStats.numPlots <= 0 ) return false;
    var vsiz = plotStats.maxX - plotStats.minX;
    var overzoom = vsiz < plotParams.numPoints;
    if( binNum >= 0 && zoomIn ) {
      setLoadChromRange(binNum,binNum);
    } else if( plotStats.zoomChrom ) {
      // set zoom out range by limits of current view
      var chr = baseRangeParam.chrom;
      var srt = baseRangeParam.pos_srt;
      var end = baseRangeParam.pos_end;
      // exception: return to previous view if in over-zoom
      if( overzoom ) {
        plotStats.minX = plotStats.baseLbins;
        plotStats.maxX = plotStats.numPoints-plotStats.baseRbins;
        setViewXtitle();
        updatePlot();
        return true;
      } else if( binNum >= 0 && !zoomIn ) {
        // zoom out from the center of the selected bin
        var csrt = dataTable[binNum][DataField.pos_start];
        var cend = dataTable[binNum][DataField.pos_end];
        var csiz = 0.5 * vsiz * plotStats.baseBinSize;
        csrt = cend = (csrt+cend) >> 1;
        csrt -= csiz;
        cend += csiz-1;
        if( csrt > 0 && cend <= plotStats.chromLength ) {
          chr = dataTable[binNum][DataField.contig_id];
          srt = csrt;
          end = cend;
        }
      }
      var siz = dblclickUnzoomFac*(end-srt+1);
      srt = Math.floor(0.5*(end+srt-siz));
      end = srt + siz - 1;
      zoomDataRange(chr,srt,end);
    } else if( plotStats.zoomInActive ) {
      $("#RC-unzoomToggle").click();
    } else if( plotStats.chromsInView < plotStats.totalChroms ) {
      // No partial zoom out with many contigs
      unzoomData();
    }
    return true;
  }

  function zoomViewToBinRange(binSrt,binEnd) {
    // correct for selections focused beyond end of last bin
    var nbin = binEnd - binSrt;
    if( binSrt < 0 ) {
      binSrt = 0;
      binEnd = binSrt + nbin;
    }
    if( binEnd >= plotStats.numPoints ) {
      binEnd = plotStats.numPoints - 1;
      binSrt = binEnd - nbin;
      if( binSrt < 0 ) binSrt = 0;
    }
    if( binSrt == binEnd ) {
      return zoomViewOnBin(binSrt,true);
    }
    // deal with range zooms across multiple and single contig views
    if( plotStats.multiChrom ) {
      if( setLoadChromRange(binSrt,binEnd) ) return;
    }
    // check for zoom on a chromosome view
    if( plotStats.binnedBases ) {
      // ensure only the middle chromosome is selected
      if( baseRangeParam.chrom === '' ) {
        var midBin = Math.round(0.5*(binSrt+binEnd));
        var chr = dataTable[midBin][DataField.contig_id];
        while( dataTable[binSrt][DataField.contig_id] != chr ) ++binSrt;
        while( dataTable[binEnd][DataField.contig_id] != chr ) --binEnd;
      }
      var chr = dataTable[binSrt][DataField.contig_id];
      var srt = dataTable[binSrt][DataField.pos_start];
      var end = dataTable[binEnd][DataField.pos_end];
      zoomDataRange(chr,srt,end);
    } else {
      // recenter at max zoom, or zoom out at over zoom
      var clip = binEnd - binSrt + 1;
      if( clip >= plotStats.minNumPoints ) {
        plotStats.minX = options.xaxis.min = binSrt;
        plotStats.maxX = options.xaxis.max = binEnd + 1;
      } else {
        var diff = 0.5 * (plotStats.minNumPoints - clip);
        plotStats.minX = Math.floor(0.5+binSrt-diff);
        plotStats.maxX = plotStats.minX + plotStats.minNumPoints;
        if( plotStats.minX < 0 ) {
          plotStats.maxX -= plotStats.minX;
          plotStats.minX = 0;
        }
        if( plotStats.maxX > plotStats.numPoints ) {
          plotStats.minX -= plotStats.maxX - plotStats.numPoints;
          plotStats.maxX = plotStats.numPoints;
        }
        binSrt = plotStats.minX;
        binEnd = plotStats.maxX - 1;
      }
      // determine if re-load is not required - true overzoom mode
      if( plotStats.zoomInActive || binEnd-binSrt+1 < plotParams.numPoints ) {
        setViewXtitle();
        checkZoomInOption();
        updatePlot();
      } else {
        var chr = dataTable[binSrt][DataField.contig_id];
        var srt = dataTable[binSrt][DataField.pos_start];
        var end = dataTable[binEnd][DataField.pos_end];
        zoomDataRange(chr,srt,end);
      }
    }
  }

  function zoomDataRange(chr,srt,end) {
    setContig(chr);
    var siz = end - srt;
    if( srt < 1 ) srt = 1;
    end = srt + siz;
    if( end > plotStats.chromLength ) {
      end = plotStats.chromLength;
      srt = end - siz;
      if( siz < 0 ) siz = 1;
    }
    baseRangeParam.maxrows = plotStats.numPoints = plotParams.numPoints;
    plotStats.zoomInActive = false;
    plotStats.zoomChrom = setContigRange(srt+'-'+end);
    zoomData();
  }

  // zoom view to a region within a single contig
  function zoomData() {
    // expand requested plot area to over-load for view panning
    if( plotStats.zoomChrom ) {
      var numPoints = plotStats.maxX - plotStats.minX;
      if( numPoints > plotStats.maxSideBars ) numPoints = plotStats.maxSideBars;
      var binsize = (baseRangeParam.pos_end - baseRangeParam.pos_srt + 1) / baseRangeParam.maxrows;
      var lbins = parseInt( (baseRangeParam.pos_srt-1)/binsize );
      if( lbins > numPoints ) lbins = numPoints;
      var rbins = parseInt( (plotStats.chromLength-baseRangeParam.pos_end)/binsize );
      if( rbins > numPoints ) rbins = numPoints;
      plotStats.baseLbins = lbins;
      plotStats.baseRbins = rbins;
      baseRangeParam.pos_srt -= parseInt(0.5+lbins*binsize);
      baseRangeParam.pos_end += parseInt(rbins*binsize);
      baseRangeParam.maxrows += lbins + rbins;
    } else {
      plotStats.baseLbins = plotStats.baseRbins = 0;
    }
    loadData();
    updatePlot();
  }

  // return view whole reference or whole contig view
  function unzoomData() {
    if( baseRangeParam.chrom !== '' ) {
      plotStats.numPoints = plotParams.numPoints;
    } else {
      plotStats.numPoints = plotStats.totalChroms < plotParams.numPoints ? plotStats.totalChroms : plotParams.numPoints;
    }
    plotStats.baseLbins = plotStats.baseRbins = 0;
    plotStats.chromLbins = plotStats.chromRbins = 0;
    plotStats.chromSrtNum = 1;
    plotStats.chromsInView = plotStats.totalChroms;
    plotStats.zoomChrom = false;
    baseRangeParam.pos_srt = 0;
    baseRangeParam.pos_end = 0;
    baseRangeParam.maxrows = plotStats.numPoints;
    contigRangeParam.binsize = plotStats.totalChroms / plotStats.numPoints;
    if( contigRangeParam.binsize < 1 ) contigRangeParam.binsize = 1;
    loadData();
    updatePlot();
  }

  function noZoomOut() {
    if( plotStats.multiChrom ) {
      return plotStats.chromsInView == plotStats.totalChroms && !plotStats.minX;
    }
    return !plotStats.zoomChrom;
  }

  function updateScrollBar() {
    if( plotStats.sliderMotive ) return;
    var slider = $('#RC-slider');
    if( noZoomOut() ) return slider.hide();
    // if looking at a window of the whole view (formally overzoom mode) then this is the slide window
    plotStats.sliderShift = 0;
    if( plotStats.maxX - plotStats.minX < plotStats.numPoints ) {
      plotStats.sliderScale = plotStats.numPoints - plotStats.maxX + plotStats.minX;
      plotStats.sliderRfPos = plotStats.minX;
    } else {
      // allow sliding beyond the data currently loaded
      var srtBin = parseInt(0.5+(dataTable[0][DataField.pos_start]-1)/plotStats.baseBinSize);
      var endBin = parseInt(0.5+plotStats.numPoints*plotStats.chromLength/plotStats.basesInView)-plotStats.numPoints;
      var dlt = plotStats.numPoints-1;
      var drt = srtBin+dlt < endBin ? dlt : endBin-srtBin;
      if( dlt > srtBin ) dlt = srtBin;
      plotStats.sliderScale = dlt+drt;
      plotStats.sliderRfPos = dlt;
    }
    slider.css('width',(plotObj.width()+2)+'px');
    slider.css('margin-left',(plotObj.getPlotOffset().left-2)+'px');
    slider.slider( "option", "max", plotStats.sliderScale );
    slider.slider( "option", "value", plotStats.sliderRfPos );
    slider.show();
  }

  $('#RC-slider').on( 'slide', function(e,u) {
    plotStats.sliderShift = u.value - plotStats.sliderRfPos;
    plotStats.sliderMotive = true;
    updatePlot();
  });

  $('#RC-slider').on( 'slidestop', function(e,u) {
    var shift = plotStats.sliderShift;
    plotStats.sliderMotive = false;
    plotStats.sliderShift = 0;
    if( shift == 0 ) return;
    zoomViewToBinRange( plotStats.minX+shift, plotStats.maxX+shift-1 );
  });

  function loadData() {
    loadTSV();
    updatePlotStats();
    checkZoomInOption();
  }

  function loadTSV() {
    var noTargets = false;
    var maxNumFields = 0;
    var pspvars = baseRangeParam;
    var srcf = 'lifechart/region_coverage.php3';
    if( plotStats.zoomInActive ) {
      srcf = wgncovFile;
    } else if( baseRangeParam.chrom === '' ) {
      var nlt = parseInt(0.5+plotStats.chromLbins*contigRangeParam.binsize);
      var nrt = parseInt(0.5+plotStats.chromRbins*contigRangeParam.binsize);
      contigRangeParam.startline = plotStats.chromSrtNum - nlt;
      contigRangeParam.numlines = plotStats.chromsInView + nlt + nrt;
      pspvars = contigRangeParam;
      // use whole contig file if available - for initial PDF conversion (not performance)
      srcf = (chrcovFile !== "" && nlt+nrt == 0) ? chrcovFile : "lifechart/fileslice.php3";
    }
    dataTable = [];
    $('#RC-message').text('Loading...');
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get( srcf, pspvars, function(mem) {
      var lines = mem.split("\n");
      $.each(lines, function(n,row) {
        var fields = $.trim(row).split('\t');
        if( n == 0 ) {
          fieldIds = fields;
          noTargets = fields.length <= 6;
          if( noTargets && plotParams.showTargets ) {
            if( plotParams.showTargets ) disableShowTargets();
            maxNumFields = fields.length;
          } else {
            // always load targets from file for zoomIn mode (as no reload for showTargets toggle)
            maxNumFields = (plotStats.zoomInActive || plotParams.showTargets) && fields.length > 7
              ? fields.length - baseRangeParam.annonumflds : fields.length;
          }
          if( fields.length < 3 ) {
            $('#RC-message').text('An error occurred while loading from server.');
            return false;
          }
          if( fields[0].substr(0,5).toLowerCase() == 'error' ) alert(row);
        } else if( fields[0] != "" ) {
          // important to convert numeric fields to numbers for performance
          var nf = fields.length < maxNumFields ? fields.length : maxNumFields;
          for( var i = 1; i < nf; ++i ) { fields[i] = +fields[i]; }
          if( noTargets ) fields[5] = fields[6] = 0;
          dataTable.push( fields );
        }
      });
    }).error(function(){
      $('#RC-message').text('An error occurred while loading from server.');
    }).success(function(){
      $('#RC-message').text('');
    });
  }

  function updatePlotStats() {
    plotStats.numPoints = dataTable.length;
    plotStats.numFields = fieldIds.length;
    plotStats.onTargets = !wholeGenome && fieldIds.length > 5;
    plotStats.multiChrom =  baseRangeParam.chrom === '';
    plotStats.binnedChroms = false;
    plotStats.minX = 0;
    plotStats.maxX = plotStats.numPoints;
    if( plotStats.numPoints == 0 ) return;
    if( !plotStats.zoomInActive && plotStats.multiChrom ) {
      plotStats.binnedChroms = contigRangeParam.binsize > 1;
      plotStats.minX = parseInt(plotStats.chromLbins);
      plotStats.maxX -= parseInt(plotStats.chromRbins);
    } else {
      plotStats.minX = plotStats.baseLbins;
      plotStats.maxX -= plotStats.baseRbins;
    }
    setViewXtitle();
    plotStats.baseBinSize = plotStats.basesInView / (plotStats.maxX - plotStats.minX);
    plotStats.binnedBases = !plotStats.multiChrom;
    if( plotStats.baseBinSize < 1.0000001 ) {
      plotStats.binnedBases = false;
      plotStats.baseBinSize = 1;
    }
    plotStats.minNumPoints = plotStats.numPoints < def_minPoints ? plotStats.numPoints : def_minPoints;
  }

  // count contigs and bases in view and set x-axis range title
  function setViewXtitle() {
    var numChr = 0, numRep = 0, lastChr = "";
    for( var i = plotStats.minX; i < plotStats.maxX; ++i ) {
      if( plotStats.binnedChroms ) {
        numChr += dataTable[i][DataField.pos_start];
        numRep += dataTable[i][DataField.pos_end];
      } else {
        if( dataTable[i][DataField.contig_id] != lastChr ) {
          ++numChr;
          lastChr = dataTable[i][DataField.contig_id];
        }
        numRep += dataTable[i][DataField.pos_end] - dataTable[i][DataField.pos_start] + 1;
      }
    }
    var chr1 = dataTable[plotStats.minX][DataField.contig_id];
    var str1 = dataTable[plotStats.minX][DataField.pos_start];
    var chrN = dataTable[plotStats.maxX-1][DataField.contig_id];
    var endN = dataTable[plotStats.maxX-1][DataField.pos_end];
    plotStats.chromsInView = numChr;
    plotStats.basesInView = numRep;
    if( plotStats.binnedChroms ) {
      var i = chr1.indexOf(' - ');
      if( i >= 0 ) chr1 = chr1.substr(0,i);
      i = chrN.indexOf(' - ');
      if( i >= 0 ) chrN = chrN.substr(i+3);
    }
    plotStats.xTitle = chr1;
    if( chr1 !== chrN ) {
      plotStats.xTitle = chr1 + ' - ' + chrN;
    } else if( str1 > 1 || endN < plotStats.chromLength ) {
      plotStats.xTitle += ': ' + commify(str1) + ' - ' + commify(endN);
    }
    plotStats.xTitle += '  (' + (chr1 !== chrN ? commify(numChr) + ' contigs, ' : '');
    plotStats.xTitle += commify(plotStats.basesInView) + ' bases)';
    if( chr1 == chrN ) {
      baseRangeParam.pos_srt = str1;
      baseRangeParam.pos_end = endN;
      $("#RC-chromRange").val(commify(str1)+'-'+commify(endN));
    } else {
      $("#RC-chromRange").val('');
    }
    var pointsInView = plotStats.maxX - plotStats.minX ;
    if( plotParams.numPoints != pointsInView ) {
      $('#RC-numBars').text('('+pointsInView+')');
    } else {
      $('#RC-numBars').text('');
    }
  }

  function updatePlot() {
    plotData = [];
    if( plotStats.numFields <= 1 ) return;
    options = {
      grid: {minBorderMargin:0, hoverable:true, clickable:true, backgroundColor:"#F8F8F8"},
      selection: {mode:plotParams.zoomMode == 2 ? "xy" : "x"},
      legend: {position:plotParams.barAxis == 0 ? "nw" : "sw"},
      series: {axis:1, bars:{show:true,align:"left"}, line:{show:false}},
      xaxis: {ticks:0, tickLength:0, axisLabel:plotStats.xTitle, axisLabelFontSizePixels:18,
        min:plotStats.minX+plotStats.sliderShift, max:plotStats.maxX+plotStats.sliderShift },
      yaxis: {tickFormatter:absFormat, axisLabelFontSizePixels:16},
      xaxes: {}, yaxes: []
    };
    var nplot = 0;
    var d1 = [];
    var d2 = [];
    var d3 = [];
    var d4 = [];
    var ymin = 0, ymax = 0;
    var pmin = 0, pmax = 0;
    var binned = plotStats.binnedBases;
    var f_reads = DataField.fwd_reads;
    var r_reads = DataField.rev_reads;
    var f_ontarg = DataField.fwd_ont;
    var r_ontarg = DataField.rev_ont;
    var chrid = DataField.contig_id;
    var srt = DataField.pos_start;
    var end = DataField.pos_end;
    var chrView = (plotStats.multiChrom && plotStats.numPoints > 1);
    var onTargets = plotStats.onTargets;
    // Minor scroll performance improvement by only working on visible window. 
    // But needs extra coding to fix variable Y axis heights and max left scroll issue.
    var xSrt = 0; //plotStats.minX + plotStats.sliderShift;
    var xEnd = plotStats.numPoints; //plotStats.maxX + plotStats.sliderShift;
    var vSrt = plotParams.autoZoom ? plotStats.minX + plotStats.sliderShift : xSrt;
    var vEnd = plotParams.autoZoom ? plotStats.maxX + plotStats.sliderShift : xEnd;
    for( var i = xSrt; i < xEnd; ++i ) {
      var scale = plotParams.aveBase ? 1.0/(dataTable[i][end]-dataTable[i][srt]+1) : 1;
      if( plotParams.barAxis == 0 ) {
        var reads = scale * (dataTable[i][f_reads]+dataTable[i][r_reads]);
        var ontarg = scale * (dataTable[i][f_ontarg]+dataTable[i][r_ontarg]);
        if( onTargets ) d1.push( [i,ontarg] );
        d2.push( [i,reads] );
        if( i >= vSrt && i <= vEnd ) {
          if( reads > ymax ) {
            pmax = ymax;
            ymax = reads;
          } else if( reads > pmax ) {
            pmax = reads;
          } 
        }
      } else {
        var freads = dataTable[i][f_reads] * scale;
        var fontarg = dataTable[i][f_ontarg] * scale;
        var rreads = -dataTable[i][r_reads] * scale;
        var rontarg = -dataTable[i][r_ontarg] * scale;
        if( onTargets ) d1.push( [i,fontarg] );
        if( onTargets ) d2.push( [i,rontarg] );
        d3.push( [i,freads] );
        d4.push( [i,rreads] );
        if( i >= vSrt && i <= vEnd ) {
          if( freads > ymax ) {
            pmax = ymax;
            ymax = freads;
          } else if( freads > pmax ) {
            pmax = freads;
          }
          if( rreads < ymin ) {
            pmin = ymin;
            ymin = rreads;
          } else if( rreads < pmin ) {
            pmin = rreads;
          }
        }
      }
    }
    // add axis labels if in multiple contigs view and not too many labels
    if( chrView && plotStats.chromsInView <= plotStats.maxXaxisLabels ) {
      var xticks = [];
      var lastChr = '';
      var lastX = 0;
      var chrN = 0;
      var formU = '<span style="font-size:9px;line-height:1px;position:relative:top:-10px">';
      var formL = '<span style="font-size:9px;line-height:1px;padding-top:-5px"><br/>';
      var formE = '</span>';
      for( var i = 0; i < plotStats.numPoints; ++i ) {
        var chr = dataTable[i][chrid];
        if( chr != lastChr ) {
          lastChr = chr;
          var label = chr;
          var dtick = (i - lastX) * 200 / plotStats.numPoints;
          lastX = i;
          // staggered labels - cludgy as do not know actual sizes (until after drawing!)
          label = ((chrN % 2) == 1 && dtick <= 2+chr.length ? formL : formU) + chr + formE;
          xticks.push( [i+0.5,label] );
          ++chrN;
        }
      }
      options.xaxis.ticks = xticks;
      options.xaxis.tickLength = 4;
    }
    // check for outlier and avoid if specified by user
    if( ymax > def_outlierFactor*pmax || ymin < def_outlierFactor*pmin ) {
      // if not yet displayed display (unchecked first if not already checked)
      if( $('#RC-offScaleOutlierControl').css("display") == 'none' ) {
        $('#RC-offScaleOutlierControl').show();
      }
      if( $('#RC-offScaleOutlier').attr('checked') ) {
        if( ymax > def_outlierFactor*pmax ) {
          ymax = 1.05 * pmax;
        }
        if( ymin < def_outlierFactor*pmin ) {
          ymin = 1.05 * pmin;
        }
      }
    } else {
      // if no outlier, remove option and reset to unchecked
      $('#RC-offScaleOutlierControl').hide();
      $('#RC-offScaleOutlier').attr('checked', (plotParams.offScaleOutlier = false) );
    }
    ymin = roundAxis(ymin);
    ymax = roundAxis(ymax);
    if( plotStats.zoomChrom && !plotParams.resetYScale ) {
      // always adjust zoom if max/min increase due to re-binning
      if( ymin < plotStats.minY ) plotStats.minY = ymin;
      if( ymax > plotStats.maxY ) plotStats.maxY = ymax;
      if( plotParams.zoomMode != 1 ) {
         ymin = plotStats.minY;
         ymax = plotStats.maxY;
      }
    } else {
      plotParams.resetYScale = false;
      plotStats.minY = ymin;
      plotStats.maxY = ymax;
    }
    // prevent flot axis defaults where whole region has 0 reads
    if( ymin == 0 && ymax == 0 ) {
      ymax = def_tinyValue;
      if( plotParams.barAxis ) ymin = -def_tinyValue;
    }

    if( plotParams.barAxis == 0 ) {
      if( onTargets ) {
        plotData.push( { label: LegendLabels.offReads, color: ColorSet.offReads, data: d2 } );
        plotData.push( { label: LegendLabels.ontReads, color: ColorSet.ontReads, data: d1 } );
      } else {
        plotData.push( { label: LegendLabels.allReads, color: ColorSet.allReads, data: d2 } );
      }
    } else {
      if( onTargets ) {
        plotData.push( { label: LegendLabels.fwdOffReads, color: ColorSet.fwdOffReads, data: d3 } );
        plotData.push( { label: LegendLabels.fwdOntReads, color: ColorSet.fwdOntReads, data: d1 } );
        plotData.push( { label: LegendLabels.revOffReads, color: ColorSet.revOffReads, data: d4 } );
        plotData.push( { label: LegendLabels.revOntReads, color: ColorSet.revOntReads, data: d2 } );
      } else {
        plotData.push( { label: LegendLabels.fwdReads, color: ColorSet.fwdOffReads, data: d3 } );
        plotData.push( { label: LegendLabels.revReads, color: ColorSet.revOffReads, data: d4 } );
      }
    }
    // show targets option disabled for whole contigs view (max zoom out)
    var showTargets = (baseRangeParam.chrom === '' && !plotStats.zoomInActive) ? false : plotParams.showTargets;
    // add (7%) margin to top of plot (for annotation)
    var cspy = showTargets ? 0.07*(ymax-ymin) : 0;
    var ytitle = plotParams.aveBase ? "Base Read Depth" : "Total Base Reads";
    options.yaxes.push( {position:"left", axisLabel:ytitle, min:ymin, max:ymax+cspy} );
    options.legend.show = plotParams.showLegend;
    plotStats.tooltipZero = 0.01*(ymax-ymin);
    ++nplot;

    // Add 2nd yaxis plot
    if( plotParams.overPlot > 0 ) {
      pointMap = []; // necessary to map to none overlayed bins
      var aLabel, pLabel, pColor;
      var d3 = [];
      if( plotParams.overPlot == 1 ) {
        aLabel = "Reads On-Target";
        pLabel = LegendLabels.percentOntarg;
        pColor = ColorSet.percentOntarg;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var totalReads = dataTable[i][f_reads] + dataTable[i][r_reads];
          if( totalReads <= 0 ) continue;
          var fwdReads = dataTable[i][f_ontarg] + dataTable[i][r_ontarg];
          var pcFwd = totalReads > 0 ? 100 * fwdReads / totalReads : 0;
          d3.push( [(i+0.5),pcFwd] );
          pointMap.push(i);
        }
      } else {
        aLabel = "Forward Reads";
        pLabel = LegendLabels.fwdBias;
        pColor = ColorSet.fwdBias;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var fwdReads = dataTable[i][f_reads];
          var totalReads = fwdReads + dataTable[i][r_reads];
          if( totalReads <= 0 ) continue;
          var pcFwd = totalReads > 0 ? 100 * fwdReads / totalReads : 50;
          d3.push( [(i+0.5),pcFwd] );
          pointMap.push(i);
        }
      }
      plotData.push( {
        label: pLabel, color: pColor, data: d3, yaxis: 2, bars: {show:false}, points: {show:true}, shadowSize: 0 } );
      options.yaxes.push( {position:"right", axisLabel:aLabel, min:0, max:101, tickFormatter: percentFormat} );
      options.grid.aboveData = true;
      if( plotParams.overPlot == 2 ) {
        options.grid.markings = [ {color: pColor, linewidth: 1, y2axis: {from:50,to:50}} ];
      }
      ++nplot;
    }

    plotStats.numPlots = nplot;
    hideTooltip();
    plotObj = $.plot(placeholder, plotData, options);
    canvas = plotObj.getCanvas();

    // add targets annotation bars
    if( showTargets ) {
      var font = "10px Times New Roman";
      var af = 7;  // query allows for any number of annotation fields but just target ID is implemented here
      // use physical view range for annotation
      xSrt = plotStats.minX + plotStats.sliderShift;
      xEnd = plotStats.maxX + plotStats.sliderShift;
      var cpos = plotObj.pointOffset( {x:xSrt,y:ymin} );
      var dx = cpos.left - 1;
      var cy = cpos.top - plotObj.height();
      var lx = dx + plotObj.getAxes().xaxis.p2c(xSrt);
      var ltxt = dataTable[xSrt].length <= af ? '' : dataTable[xSrt][af];
      var rdot = (ltxt !== '');
      var binClick = xSrt;
      for( var bin = ++xSrt; bin <= xEnd; ++bin ) {
        var atxt = (bin == xEnd || dataTable[bin].length <= af) ? '' : dataTable[bin][af];
        if( atxt !== ltxt ) {
          var cx = dx + plotObj.getAxes().xaxis.p2c(bin);
          if( ltxt !== '' ) {
            var wd = cx-lx;
            var tx = shortenTxtToWidth(wd,ltxt,font,rdot);
            var col = targetBarColorByTxt(ltxt);
            rdot = false;
            var elm = $("<div bin='"+binClick+"' title='"+ltxt+"' style='" +
              "position:absolute;border:1px solid #880;z-index:10;text-align:center;cursor:pointer;" +
              "left:"+lx+"px;top:"+cy+"px;width:"+wd+"px;background:"+col+";font:"+font+"'>"+tx+"</div>");
            placeholder.append(elm);
            elm.click(targetBarClick);
          }
          lx = cx;
          ltxt = atxt;
          binClick = bin;
        }
      }
    }
    updateScrollBar();
  }

  function getTextWidth(txt,font) {
    var fnt = font || '10px arial';
    var elm = $("<div style='position:absolute;white-space:nowrap;visibility:hidden;font:"+fnt+"'>"+txt+"</div>");
    elm.appendTo($('body'));
    var w = elm.width();
    elm.remove();
    return w;
  }

  function shortenTxtToWidth(width,txt,font,rdot) {
    var tw = getTextWidth(txt,font);
    if( tw <= width ) return txt;
    var ln = parseInt(txt.length * width / tw) - 1;
    if( ln < 0 ) return ".";
    return rdot ? ".."+txt.substr(txt.length-ln) : txt.substr(0,ln)+"..";
  }

  function targetBarColorByTxt(txt) {
    var ncom = (txt.match(/,/g)||[]).length;
    if( ncom > 2 ) ncom = 2;
    return ["#FF0","#EC2","#DA4"][ncom]+";opacity:0.7";
  }

  $('#RC-export').click( function() {
    // determine binning strategy for report
    var i = plotStats.xTitle.indexOf('  (');
    var contigs = plotStats.xTitle.substr(0,i);
    var spans = plotStats.xTitle.substring(i+3,plotStats.xTitle.length-1);
    var lead = contigs.indexOf(':') < 0 ? "Contig"+(plotStats.multiChrom ? "s" : "") : "Region";
    var numbins = plotStats.maxX - plotStats.minX;
    spans += " in "+commify(numbins)+" bins"
    var bedDisable = "";
    var wstr = "";
    if( plotStats.multiChrom ) {
      if( plotStats.zoomInActive ) {
        wstr = plotStats.zoomInActive ? "Apportioned by relative contig lengths" : "Whole contig lengths";
      } else if( plotStats.binnedChroms ) {
        var nchr = plotStats.chromsInView / numbins;
        var ichr = parseInt(nchr);
        if( ichr != nchr ) ichr = ichr + " or " + (ichr+1);
        wstr = "Multiple ("+ichr+") whole contig lengths";
        bedDisable = 'style="display:none"'; //'disabled="disabled"';
      } else {
        wstr = "Whole contig lengths";
      }
    } else {
      var binint = parseInt(plotStats.baseBinSize);
      var binfrc = plotStats.baseBinSize - binint;
      if( binfrc == 0 ) {
        wstr = commify(binint)+" base"+(binint > 1 ? "s" : "");
      } else if( plotStats.baseBinSize > 10*cbcsize ) {
        // to account CBC file pre-binned kick-in test used in code
        var numcbcs = plotStats.baseBinSize / cbcsize;
        binint = parseInt(numcbcs);
        if( numcbcs > binint ) ++binint;
        binint *= cbcsize;
        var sbins =  plotStats.basesInView - binint * (numbins-1);
        wstr = commify(binint)+" bases ("+commify(sbins)+" in last bin)";
      } else {
        var sbins = plotStats.basesInView - binint * numbins;
        var sstr = "more";
        if( binfrc >= 0.5 ) {
          sstr = "less";
          ++binint;
          sbins = numbins - sbins;
        }
        wstr = "~"+commify(binint)+" bases (1 "+sstr+" in "+commify(sbins)+" bins)";
      }
    }
    // fill in dialog
    var $content = $('#RC-dialog-content');
    $content.html(lead+' in view: '+contigs+'<br/>Spanning: '+spans+'<br/>Bin width: '+wstr+'<br/><br/>' );
    $content.append('<p>\
      <input type="radio" name="TS-exportTool" id="RC-ext1" value="table" checked="checked"/>\
        <label for="RC-ext1">Download as tsv table file.</label><br/>\
      <input type="radio" name="TS-exportTool" id="RC-ext2" value="bed" '+bedDisable+'"+/>\
        <label for="RC-ext2" '+bedDisable+'>Download as a 3-column bed file.</label></p>' );
    $('#RC-exportOK').show();
    // open dialog over masked out table
    var pos = $('#RC-export').offset();
    var x = pos.left+6+($('#RC-export').width()-$('#RC-dialog').width())/2;
    var y = pos.top+$('#RC-export').height()+8;
    $('#RC-dialog').css({ left:x, top:y });
    pos = $('#RC-chart').offset();
    var hgt = $('#RC-chart').height()+4; // extra for borders
    var wid = $('#RC-chart').width()+4;
    $('#RC-mask').css({ left:pos.left, top:pos.top, width:wid, height:hgt });
    $('#RC-mask').show();
    $('#RC-dialog').show();
  });

  $('#RC-exportOK').click(function(e) {
    $('#RC-dialog').hide();
    $('#RC-mask').hide();
    // the following doesn't work when including DOC and RCC code and but not displaying (exiting immediately)
    var op = $("input[@name='TS-exportTool']:checked").val();
    if( op == "table" ) {
      exportTSV(true);
    } else if( op == "bed" ) {
      exportTSV(false);
    }
  });

  function exportTSV(toTable) {
    // choose suitable output file name
    var outfile = "RCC_download";
    if( chrcovFile ) {
      outfile = chrcovFile;
    } else if( wgncovFile ) {
      outfile = wgncovFile;
    }
    var i = outfile.lastIndexOf('/');
    if( i >= 0 ) outfile = outfile.substr(i+1);
    i = outfile.indexOf('.');
    if( i >= 0 ) outfile = outfile.substr(0,i);
    outfile += ".ref.cov.xls";
    // HTML5 download facility not supported for IE9 or less.
    // Browser detection disabled on TS and form/post not working at time of writing.
    // Hence, solution is to just GET and PHP to (re)extract and process the data already at hand in this script.
    if( baseRangeParam.chrom === '' ) {
      var pspvars = contigRangeParam;
      var i = baseRangeParam.bbcfile.lastIndexOf('/');
      var fpath = i < 0 ? "" : baseRangeParam.bbcfile.substring(0,i+1);
      var fname, srtline, binsize, numlines, binsrt, binend;
      if( plotStats.zoomInActive ) {
        fname = wgncovFile;
        srtline = plotStats.minX + 1;
        binsize = 1;
        numlines = plotStats.maxX - plotStats.minX;
        binsrt = binend = 0;
      } else {
        // use view of same region as last (over) loaded to ensure data matches view exactly
        fname = chrcovFile;
        srtline = contigRangeParam.startline;
        binsize = contigRangeParam.binsize;
        numlines = contigRangeParam.numlines;
        binsrt = plotStats.minX + 1;
        binend = plotStats.maxX;
      }
      window.open( "lifechart/fileslice.php3"+ "?filename="+fpath+fname+"&outfile="+outfile+
        "&startline="+srtline+"&numlines="+numlines+"&binsize="+binsize+"&binsrt="+binsrt+"&binend="+binend+
        "&bedcoords="+(toTable ? 0 : 1)+"&headlines="+(toTable ? 1 : -1)+"&numfields="+(toTable ? 7 : 3) );
    } else {
      var pos_srt, pos_end, srt_bin, end_bin, maxrows;
      if( plotStats.baseBinSize <= 1 ) {
        // single base extract allows for over zoom
        pos_srt = baseRangeParam.pos_srt;
        pos_end = baseRangeParam.pos_end;
        maxrows = pos_end - pos_srt + 1;
        srt_bin = end_bin = 0;
      } else {
        // recalculate full loading so that data matches view exactly
        var numBins = baseRangeParam.maxrows - plotStats.baseLbins - plotStats.baseRbins;
        var binsize = (baseRangeParam.pos_end - baseRangeParam.pos_srt + 1) / numBins;
        pos_srt = baseRangeParam.pos_srt - parseInt(0.5+plotStats.baseLbins*binsize);
        pos_end = baseRangeParam.pos_end + parseInt(plotStats.baseRbins*binsize);
        maxrows = baseRangeParam.maxrows;
        srt_bin = plotStats.minX;
        end_bin = plotStats.maxX;
      }
      window.open( "lifechart/region_coverage.php3"+
        "?bbcfile="+baseRangeParam.bbcfile+"&annofile="+baseRangeParam.annofile+"&maxrows="+maxrows+
        "&chrom="+baseRangeParam.chrom+"&pos_srt="+pos_srt+"&pos_end="+pos_end+
        "&outfile="+outfile+"&srt_bin="+srt_bin+"&end_bin="+end_bin+
        "&options="+(toTable ? "" : "-bl") );
    }
  }

  // start up customization attributes
  if( startShowOptions )
    $('#RC-controlpanel').show();
  if( startOutlierOffScale )
    $('#RC-offScaleOutlier').attr('checked', (plotParams.offScaleOutlier = true) );
  if( annoFile == '' )
    disableShowTargets();
  loadContigs();
  autoHideLegend();

  // automatically change initial view to full contig if only one
  // done this way to plot only once (for sake of auto PDF)
  if( plotStats.totalChroms == 1 ) {
    $("#RC-unzoomToggle").click();
  } else {
    unzoomData();
  }
  if( plotStats.totalChroms == 1 || plotStats.totalChroms > plotStats.maxXaxisLabels ) {
    wgncovFile = '';
  }
  setUnzoomTitle( wgncovFile !== '' );
  // collapse view after EVERYTHING has been drawn in open chart (to avoid flot issues)
  if( startCollapsed ) {
    $("#RC-collapsePlot").attr("class","ui-icon ui-icon-triangle-1-s");
    $("#RC-collapsePlot").attr("title","Expand view");
    $('#RC-controlpanel').hide();
    $('.RC-shy').hide();
    $('#RC-chart').resizable('destroy');
    $('#RC-noncanvas').hide();
  }
});
